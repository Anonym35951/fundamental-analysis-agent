from sqlalchemy.orm import Session

from api.models.stripe_event import StripeEvent


def is_stripe_event_processed(db: Session, event_id: str) -> bool:
    existing_event = (
        db.query(StripeEvent)
        .filter(StripeEvent.event_id == event_id)
        .first()
    )
    return existing_event is not None


def mark_stripe_event_processed(db: Session, event_id: str, event_type: str) -> StripeEvent:
    stripe_event = StripeEvent(
        event_id=event_id,
        event_type=event_type,
    )
    db.add(stripe_event)
    db.commit()
    db.refresh(stripe_event)
    return stripe_event