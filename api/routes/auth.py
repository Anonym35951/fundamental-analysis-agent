import hashlib
import logging
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from api.core.rate_limit import limiter

from api.schemas.user import UserCreate, UserProfileUpdateRequest, UserResponse
from api.schemas.auth import (
    TokenResponse,
    ChangePasswordRequest,
    DeleteAccountRequest,
    ForgotPasswordRequest,
    ResendVerificationPublicRequest,
    ResetPasswordRequest,
    VerifyEmailRequest,
)
from api.crud.user import (
    create_user,
    get_user_by_email,
    get_user_by_email_or_username,
    get_user_by_username,
    update_user_profile,
)
from api.core.dependencies import get_db, get_current_user
from api.core.config import settings
from api.core.security import (
    verify_password,
    create_access_token,
    hash_password,
)
from api.models.user import User
from api.services.event_service import log_event
from api.services.user_service import StripeCancellationError, delete_user_account
from api.services.email_service import (
    send_email_safely,
    send_email_verification_email,
    send_password_changed_email,
    send_password_reset_email,
)

PASSWORD_RESET_TOKEN_TTL_MINUTES = 30
EMAIL_VERIFICATION_TOKEN_TTL_HOURS = 24

# Versionsstempel der aktuell gueltigen Rechtstexte. Bei inhaltlicher
# Aenderung von ToS/Datenschutz einfach den String hier hochzaehlen -
# ermoeglicht spaeter (falls gewuenscht) eine Auswertung, wer welcher
# Fassung zugestimmt hat. Kein erzwungenes Re-Consent-Flow heute.
CURRENT_TERMS_VERSION = "2026-07"
CURRENT_PRIVACY_VERSION = "2026-07"

router = APIRouter(prefix="/auth", tags=["auth"])

logger = logging.getLogger(__name__)


def _issue_email_verification_token(db: Session, user: User) -> str:
    """Erzeugt einen frischen Verifizierungs-Token (Muster wie Passwort-Reset:
    roher Token nur im Mail-Link, DB speichert den SHA-256-Hash)."""
    raw_token = secrets.token_urlsafe(32)
    user.email_verification_token_hash = hashlib.sha256(
        raw_token.encode()
    ).hexdigest()
    user.email_verification_expires = datetime.utcnow() + timedelta(
        hours=EMAIL_VERIFICATION_TOKEN_TTL_HOURS
    )
    db.commit()
    return raw_token


def _send_verification_email(background_tasks: BackgroundTasks, user: User, raw_token: str) -> None:
    verify_link = f"{settings.FRONTEND_URL}/verify-email?token={raw_token}"
    background_tasks.add_task(send_email_safely, send_email_verification_email, user.email, verify_link)


@router.post("/register", response_model=UserResponse)
@limiter.limit("3/minute")
def register(
    request: Request,
    user: UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    existing_user = get_user_by_email(db, str(user.email))
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    if get_user_by_username(db, user.username):
        raise HTTPException(status_code=400, detail="Benutzername bereits vergeben")

    created = create_user(
        db,
        str(user.email),
        user.password,
        username=user.username,
        first_name=user.first_name,
        last_name=user.last_name,
        age=user.age,
        terms_version=CURRENT_TERMS_VERSION,
        privacy_version=CURRENT_PRIVACY_VERSION,
    )

    raw_token = _issue_email_verification_token(db, created)
    _send_verification_email(background_tasks, created, raw_token)

    log_event(db, "user_registered", user_id=created.id)

    return created


@router.post("/verify-email")
@limiter.limit("10/minute")
def verify_email(
    request: Request,
    data: VerifyEmailRequest,
    db: Session = Depends(get_db),
):
    token_hash = hashlib.sha256(data.token.encode()).hexdigest()
    user = (
        db.query(User)
        .filter(User.email_verification_token_hash == token_hash)
        .first()
    )

    if (
        not user
        or not user.email_verification_expires
        or user.email_verification_expires < datetime.utcnow()
    ):
        raise HTTPException(status_code=400, detail="Token is invalid or has expired")

    user.email_verified = True
    user.email_verification_token_hash = None
    user.email_verification_expires = None
    db.commit()

    log_event(db, "email_verified", user_id=user.id)

    return {"message": "Email verified successfully"}


@router.post("/resend-verification")
@limiter.limit("3/minute")
def resend_verification(
    request: Request,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if current_user.email_verified:
        raise HTTPException(status_code=400, detail="Email is already verified")

    raw_token = _issue_email_verification_token(db, current_user)
    _send_verification_email(background_tasks, current_user, raw_token)

    return {"message": "Verification email sent"}


@router.post("/resend-verification-public")
@limiter.limit("3/minute")
def resend_verification_public(
    request: Request,
    data: ResendVerificationPublicRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Gegenstueck zu /resend-verification fuer den Fall, dass der Login
    selbst wegen fehlender Verifizierung blockiert ist (siehe /login) - ohne
    gueltigen Token kann der Nutzer die eingeloggte Variante nicht erreichen.
    Gleiche generische Antwort unabhaengig vom tatsaechlichen Kontostatus wie
    /forgot-password, um Account-Enumeration zu vermeiden."""
    generic_response = {
        "message": (
            "Falls ein Konto mit dieser Angabe existiert und noch nicht "
            "bestätigt ist, haben wir einen neuen Bestätigungslink gesendet."
        )
    }

    user = get_user_by_email_or_username(db, data.identifier)
    if not user or user.email_verified:
        return generic_response

    raw_token = _issue_email_verification_token(db, user)
    _send_verification_email(background_tasks, user, raw_token)

    return generic_response


@router.post("/login", response_model=TokenResponse)
@limiter.limit("5/minute")
def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # form_data.username ist nur der Name des OAuth2-Formularfelds - der
    # Nutzer kann hier sowohl seine Email als auch seinen (neuen) Benutzernamen
    # eingeben, beides ist gleichwertig fuers Login.
    user = get_user_by_email_or_username(db, form_data.username)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.email_verified:
        raise HTTPException(
            status_code=403,
            detail={
                "code": "EMAIL_NOT_VERIFIED",
                "message": (
                    "Deine E-Mail-Adresse ist noch nicht bestätigt. Bitte "
                    "bestätige sie über den Link in deiner Bestätigungs-Mail, "
                    "um dich einzuloggen."
                ),
            },
        )

    access_token = create_access_token(subject=str(user.id))

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


@router.patch("/profile", response_model=UserResponse)
def update_profile(
    data: UserProfileUpdateRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Optionale Profil-Nachpflege (Benutzername/Vorname/Nachname/Alter).
    Dient sowohl Bestandsnutzern (kein Zwang, rein freiwillig ueber die
    Account-Seite) als auch kuenftigen Profil-Bearbeitungen."""
    try:
        updated = update_user_profile(db, current_user, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return updated


@router.post("/onboarding-complete")
def onboarding_complete(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Markiert die gefuehrte Tour als abgeschlossen (oder uebersprungen -
    beides zaehlt als 'erledigt', damit sie nicht erneut nervt)."""
    current_user.onboarding_completed_at = datetime.utcnow()
    db.commit()
    log_event(db, "onboarding_tour_completed", user_id=current_user.id)
    return {"message": "ok"}


@router.post("/refresh", response_model=TokenResponse)
def refresh(current_user: User = Depends(get_current_user)):
    """Exchanges a still-valid token for a fresh one. Powers the frontend's
    session keep-alive (periodic background renewal while the app is open)
    and the "Sitzung verlängern" inactivity-warning flow — neither would do
    anything otherwise, since the token's expiry is fixed at issuance time
    regardless of activity."""
    access_token = create_access_token(subject=str(current_user.id))

    return {
        "access_token": access_token,
        "token_type": "bearer",
    }


@router.patch("/change-password")
def change_password(
    data: ChangePasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect",
        )

    current_user.hashed_password = hash_password(data.new_password)
    db.commit()
    db.refresh(current_user)

    # Bestätigungs-Mail asynchron nach dem Response senden, blockiert den Flow nicht.
    background_tasks.add_task(send_email_safely, send_password_changed_email, current_user.email)

    return {"message": "Password changed successfully"}


@router.post("/forgot-password")
@limiter.limit("3/minute")
def forgot_password(
    request: Request,
    data: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    generic_response = {
        "message": "If an account with this email exists, a password reset link has been sent."
    }

    user = get_user_by_email(db, str(data.email))
    if not user:
        return generic_response

    raw_token = secrets.token_urlsafe(32)
    user.password_reset_token_hash = hashlib.sha256(raw_token.encode()).hexdigest()
    user.password_reset_expires = datetime.utcnow() + timedelta(
        minutes=PASSWORD_RESET_TOKEN_TTL_MINUTES
    )
    db.commit()

    reset_link = f"{settings.FRONTEND_URL}/reset-password?token={raw_token}"
    background_tasks.add_task(send_email_safely, send_password_reset_email, user.email, reset_link)

    return generic_response


@router.post("/reset-password")
@limiter.limit("5/minute")
def reset_password(
    request: Request,
    data: ResetPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    token_hash = hashlib.sha256(data.token.encode()).hexdigest()
    user = (
        db.query(User)
        .filter(User.password_reset_token_hash == token_hash)
        .first()
    )

    if (
        not user
        or not user.password_reset_expires
        or user.password_reset_expires < datetime.utcnow()
    ):
        raise HTTPException(status_code=400, detail="Token is invalid or has expired")

    user.hashed_password = hash_password(data.new_password)
    user.password_reset_token_hash = None
    user.password_reset_expires = None
    db.commit()

    background_tasks.add_task(send_email_safely, send_password_changed_email, user.email)

    return {"message": "Password reset successfully"}


@router.post("/delete-account")
@limiter.limit("3/minute")
def delete_account(
    request: Request,
    data: DeleteAccountRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """DSGVO-Konto-Löschung (Hard Delete): kündigt ein laufendes Stripe-Abo
    sofort und löscht danach alle Nutzerdaten endgültig."""
    if not verify_password(data.password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="Current password is incorrect",
        )

    try:
        delete_user_account(db, current_user, event_metadata={"deleted_by": "self"})
    except StripeCancellationError:
        raise HTTPException(
            status_code=502,
            detail=(
                "Dein Abo konnte gerade nicht gekündigt werden. Bitte "
                "versuche es später erneut oder kontaktiere den Support."
            ),
        )

    return {"message": "Account deleted"}