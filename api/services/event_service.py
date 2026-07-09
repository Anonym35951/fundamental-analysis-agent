import logging

from sqlalchemy.orm import Session

from api.models.product_event import ProductEvent

logger = logging.getLogger(__name__)


def log_event(
    db: Session,
    event_type: str,
    user_id: int | None = None,
    metadata: dict | None = None,
) -> None:
    """Schreibt ein Produkt-Event fürs private Admin-Dashboard.

    Bewusst fehlertolerant: ein Fehler beim Event-Logging darf niemals den
    eigentlichen Request (Registrierung, Analyse-Start, Zahlung) zum Absturz
    bringen — im Zweifel fehlt nur ein Datenpunkt in der Statistik.
    """
    try:
        db.add(
            ProductEvent(event_type=event_type, user_id=user_id, event_metadata=metadata)
        )
        db.commit()
    except Exception:
        db.rollback()
        logger.exception("Failed to log product event: %s", event_type)
