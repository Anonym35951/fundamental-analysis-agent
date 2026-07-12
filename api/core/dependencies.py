from datetime import date, datetime
from api.utils.time import utcnow

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from api.core.config import settings
from api.core.database import SessionLocal
from api.models.user import User
from api.services.event_service import log_event
from api.services.user_service import try_consume_monthly_request_quota


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")
# auto_error=False: liefert None statt 401, wenn kein Token mitgeschickt wird -
# fuer Endpunkte, die sowohl eingeloggt als auch anonym erreichbar sein muessen
# (z. B. das oeffentliche Support-Formular, das bei vorhandenem Login
# zusaetzlichen User-Kontext in die Mail schreibt).
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )

    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.id == int(user_id)).first()
    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    # Limit-Prüfung und Verbrauchszählung bewusst NICHT hier: get_current_user
    # wird auch von nicht-analytischen Endpunkten (z. B. /auth/me, Account-/
    # Billing-Seiten) genutzt. Würde hier bereits 403 ausgelöst, wäre die App
    # für Free-User nach Erreichen des Limits komplett blockiert (auch
    # Dashboard/Sidebar, die /auth/me aufrufen). Tatsächliche Analyse-
    # Endpunkte nutzen stattdessen require_analysis_access (siehe unten).
    return user


def get_current_user_optional(
    token: str | None = Depends(oauth2_scheme_optional),
    db: Session = Depends(get_db),
) -> User | None:
    """Wie get_current_user, gibt aber None zurueck statt 401 auszuloesen,
    wenn kein oder ein ungueltiger Token mitgeschickt wurde."""
    if not token:
        return None
    try:
        return get_current_user(token=token, db=db)
    except HTTPException:
        return None


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    """Admin-Gate fuer private Admin-Routen (Stats + Kunden-CRM). Gleiche
    Konvention wie die admin/*-Endpoints in auth.py: plan == "admin"."""
    if current_user.plan != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


EMAIL_NOT_VERIFIED_DETAIL = (
    "E-Mail-Adresse noch nicht bestätigt. Bitte klicke auf den Link in der "
    "Bestätigungs-Mail, um Analysen zu starten."
)


def _require_verified_email(current_user: User) -> None:
    if not current_user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=EMAIL_NOT_VERIFIED_DETAIL,
        )


def _next_monthly_reset_date() -> str:
    """Erster Tag des Folgemonats (UTC). Free-Kontingente laufen kalendermonat-
    weise und werden "on-read" beim nächsten Quota-Check zurückgesetzt
    (try_consume_monthly_request_quota, usage_period_start) — dieses Datum
    ist rein informativ fürs Frontend und unabhängig vom gespeicherten
    Zählerstand immer korrekt, da beide auf demselben Kalendermonat basieren."""
    now = utcnow()
    if now.month == 12:
        reset_day = date(now.year + 1, 1, 1)
    else:
        reset_day = date(now.year, now.month + 1, 1)
    return reset_day.isoformat()


def _quota_exceeded_detail() -> dict:
    return {
        "code": "QUOTA_EXCEEDED",
        "message": "Monatliches Analyse-Kontingent aufgebraucht. Upgrade auf Pro für unbegrenzte Analysen.",
        "reset_date": _next_monthly_reset_date(),
    }


def require_analysis_access(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> User:
    """Auth-Dependency für Endpunkte, die eine Analyse tatsächlich ausführen.

    Prüft das Monatslimit für Free-User und zählt den Verbrauch erst hier,
    damit reine Status-Abfragen (progress/result) und nicht-analytische
    Endpunkte (z. B. /auth/me) das Kontingent nicht zusätzlich belasten oder
    fälschlich blockiert werden.
    """
    _require_verified_email(current_user)

    if current_user.plan == "free":
        if not try_consume_monthly_request_quota(db, current_user.id, units=1):
            log_event(db, "quota_exceeded", user_id=current_user.id, metadata={"units": 1})
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=_quota_exceeded_detail(),
            )
    return current_user


def require_analysis_access_for_units(db: Session, current_user: User, units: int) -> None:
    """Wie require_analysis_access, aber für Endpunkte, die mehrere
    Analyse-Einheiten in einem Request verbrauchen (z. B. N ausgewählte
    Kennzahlen × 1 Unternehmen pro Custom-Analysis-/ComparePage-Aufruf).
    Muss explizit in der Route aufgerufen werden, nachdem die Anzahl der
    angeforderten Kennzahlen bekannt ist — nicht als Depends() nutzbar.
    """
    _require_verified_email(current_user)

    if current_user.plan == "free":
        if not try_consume_monthly_request_quota(db, current_user.id, units=units):
            log_event(db, "quota_exceeded", user_id=current_user.id, metadata={"units": units})
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=_quota_exceeded_detail(),
            )