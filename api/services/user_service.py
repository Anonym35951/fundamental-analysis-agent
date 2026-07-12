import logging
from datetime import date, datetime
from api.utils.time import utcnow

import stripe
from sqlalchemy import and_, case, or_, update
from sqlalchemy.orm import Session

from api.models.analysis_history import AnalysisHistory
from api.models.custom_analysis_definition import CustomAnalysisDefinition
from api.models.customer_note import CustomerNote
from api.models.favorite import Favorite
from api.models.user import User
from api.services.event_service import log_event
from api.services.stripe_service import cancel_subscription_immediately

logger = logging.getLogger(__name__)


class StripeCancellationError(Exception):
    """Stripe-Kündigung schlug fehl (kein InvalidRequestError) — das Konto
    darf nicht gelöscht werden, sonst liefe ein Abo auf ein gelöschtes
    Konto weiter."""


def delete_user_account(
    db: Session,
    user: User,
    *,
    event_metadata: dict | None = None,
) -> None:
    """Hard Delete eines Nutzerkontos: kündigt ein laufendes Stripe-Abo
    sofort und löscht danach alle Nutzerdaten endgültig. Gemeinsame Logik
    für den Self-Service-DSGVO-Flow (auth.py) und die Admin-CRM-Löschung
    (admin_customers.py) — Zugriffsprüfungen (Passwort bzw. require_admin)
    bleiben Sache der Routen.

    Raises:
        StripeCancellationError: Stripe-Kündigung fehlgeschlagen; es wurde
        nichts gelöscht.
    """
    if user.stripe_subscription_id:
        try:
            cancel_subscription_immediately(user.stripe_subscription_id)
        except stripe.error.InvalidRequestError:
            # Abo existiert bei Stripe nicht (mehr) — Löschung fortsetzen.
            logger.warning(
                "Stripe subscription not found during account deletion: user_id=%s",
                user.id,
            )
        except Exception as exc:
            logger.exception(
                "Stripe cancellation failed during account deletion: user_id=%s",
                user.id,
            )
            raise StripeCancellationError() from exc

    user_id = user.id

    # Vor dem Löschen protokollieren: product_events.user_id ist ON DELETE
    # SET NULL, das Event selbst bleibt also für die Statistik erhalten.
    log_event(db, "account_deleted", user_id=user_id, metadata=event_metadata)

    # Reihenfolge wichtig: analysis_history.user_id hat kein ON DELETE CASCADE.
    db.query(AnalysisHistory).filter(AnalysisHistory.user_id == user_id).delete()
    db.query(Favorite).filter(Favorite.user_id == user_id).delete()
    db.query(CustomAnalysisDefinition).filter(
        CustomAnalysisDefinition.user_id == user_id
    ).delete()
    db.query(CustomerNote).filter(CustomerNote.user_id == user_id).delete()
    db.query(User).filter(User.id == user_id).delete()
    db.commit()

    logger.info("Account deleted: user_id=%s", user_id)


def _current_period_start() -> date:
    """Erster Tag des laufenden Kalendermonats (UTC) — Referenzpunkt für den
    on-read-Reset des Free-Kontingents in try_consume_monthly_request_quota."""
    now = utcnow()
    return date(now.year, now.month, 1)


def get_user_for_update(db: Session, user_id: int) -> User | None:
    return (
        db.query(User)
        .filter(User.id == user_id)
        .with_for_update()
        .first()
    )


def update_user_plan(
    db: Session,
    user_id: int,
    new_plan: str,
    stripe_customer_id: str | None = None,
    stripe_subscription_id: str | None = None,
    billing_interval: str | None = None,
    stripe_price_id: str | None = None,
) -> User | None:
    user = get_user_for_update(db, user_id)
    if user is None:
        return None

    user.plan = new_plan

    if new_plan == "free":
        user.monthly_request_limit = 50
        user.billing_interval = None
        user.stripe_price_id = None
    else:
        user.monthly_request_limit = None

        if billing_interval is not None:
            user.billing_interval = billing_interval

        if stripe_price_id is not None:
            user.stripe_price_id = stripe_price_id

    if stripe_customer_id is not None:
        user.stripe_customer_id = stripe_customer_id

    if stripe_subscription_id is not None:
        user.stripe_subscription_id = stripe_subscription_id
    elif new_plan == "free":
        user.stripe_subscription_id = None

    db.commit()
    db.refresh(user)
    return user


def try_consume_monthly_request_quota(db: Session, user_id: int, units: int = 1) -> bool:
    """Prüft das Free-Plan-Monatslimit und inkrementiert monthly_request_count
    atomar in einem einzigen UPDATE-Statement. Ersetzt das vorherige Lesen
    (current_user.monthly_request_count) gefolgt von einem separaten
    increment_monthly_request_count()-Schreibzugriff, das bei echt
    gleichzeitigen Requests (z. B. ComparePage feuert einen Request pro
    Unternehmen parallel) zu verlorenen Inkrementen führen konnte.

    Enthält außerdem den "on-read"-Reset für Free-User: Statt dass ein
    zeitgesteuerter Worker am 1. jedes Monats alle Zähler zurücksetzt (der
    Reset fällt aus, wenn die App am Monatsersten gerade deployed/schlafend
    ist — Render!), wird hier bei jedem Quota-Check geprüft, ob
    usage_period_start noch der aktuelle Kalendermonat ist. Falls nicht,
    zählt effective_count als 0 statt des gespeicherten Werts — alles in
    derselben atomaren UPDATE-Anweisung, damit kein separater
    Lese-dann-Schreib-Schritt nötig ist. Gilt bewusst nur für 'free': Pro-/
    Admin-User behalten ihren monthly_request_count als laufenden
    Gesamtzähler (siehe admin_stats.py / Dashboard-Widget).

    Returns:
        True  – Kontingent verbraucht (oder Nutzer ist nicht 'free', also
                 unbegrenzt); monthly_request_count wurde erhöht (ggf. nach
                 Reset auf den aktuellen Monat).
        False – hätte das Limit überschritten; nichts wurde verändert.
    """
    period_start = _current_period_start()
    is_stale_free_period = and_(
        User.plan == "free",
        or_(
            User.usage_period_start.is_(None),
            User.usage_period_start != period_start,
        ),
    )
    effective_count = case(
        (is_stale_free_period, 0),
        else_=User.monthly_request_count,
    )

    result = db.execute(
        update(User)
        .where(
            User.id == user_id,
            or_(
                User.plan != "free",
                effective_count + units <= User.monthly_request_limit,
            ),
        )
        .values(
            monthly_request_count=effective_count + units,
            usage_period_start=period_start,
        )
    )
    db.commit()
    return result.rowcount > 0


def increment_monthly_request_count(db: Session, user_id: int, count: int = 1) -> User | None:
    user = get_user_for_update(db, user_id)
    if user is None:
        return None

    user.monthly_request_count += count
    db.commit()
    db.refresh(user)
    return user


def reset_monthly_request_count(db: Session, user_id: int) -> User | None:
    user = get_user_for_update(db, user_id)
    if user is None:
        return None

    user.monthly_request_count = 0
    user.usage_period_start = _current_period_start()
    db.commit()
    db.refresh(user)
    return user


def reset_all_free_users_monthly_request_counts(db: Session) -> int:
    """Manueller Bulk-Reset für alle Free-User. Seit Einführung des
    on-read-Resets in try_consume_monthly_request_quota nicht mehr für die
    Korrektheit erforderlich (jeder Quota-Check erkennt einen veralteten
    Monat selbst) — bleibt als optionales Admin-/Cron-Werkzeug erhalten,
    z. B. um Zähler unabhängig vom nächsten Request sofort sichtbar
    zurückzusetzen (Dashboard-Anzeige)."""
    updated_rows = (
        db.query(User)
        .filter(User.plan == "free")
        .update(
            {
                User.monthly_request_count: 0,
                User.usage_period_start: _current_period_start(),
            },
            synchronize_session=False,
        )
    )
    db.commit()
    return updated_rows


def downgrade_expired_past_due_users(db: Session) -> int:
    now = utcnow()

    users = (
        db.query(User)
        .filter(
            User.billing_status == "past_due",
            User.grace_until.isnot(None),
            User.grace_until < now,
        )
        .with_for_update()
        .all()
    )

    downgraded_count = 0

    for user in users:
        user.plan = "free"
        user.billing_status = "payment_failed_canceled"
        user.billing_interval = None
        user.stripe_price_id = None
        user.current_period_end = None
        user.grace_until = None
        user.monthly_request_limit = 50
        user.stripe_subscription_id = None
        downgraded_count += 1

    if downgraded_count > 0:
        db.commit()

    return downgraded_count