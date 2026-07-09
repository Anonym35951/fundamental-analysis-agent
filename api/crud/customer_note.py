from sqlalchemy.orm import Session

from api.models.customer_note import CustomerNote


def create_note(db: Session, user_id: int, admin_author_id: int, note_text: str) -> CustomerNote:
    note = CustomerNote(user_id=user_id, admin_author_id=admin_author_id, note=note_text)
    db.add(note)
    db.commit()
    db.refresh(note)
    return note


def get_notes_for_user(db: Session, user_id: int) -> list[CustomerNote]:
    return (
        db.query(CustomerNote)
        .filter(CustomerNote.user_id == user_id)
        .order_by(CustomerNote.created_at.desc())
        .all()
    )
