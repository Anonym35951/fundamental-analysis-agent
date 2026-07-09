from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.models.favorite import Favorite


def get_favorites(db: Session, user_id: int) -> list[Favorite]:
    return (
        db.query(Favorite)
        .filter(Favorite.user_id == user_id)
        .order_by(Favorite.created_at.desc())
        .all()
    )


def add_favorite(db: Session, user_id: int, symbol: str) -> Favorite:
    normalized = symbol.strip().upper()
    existing = (
        db.query(Favorite)
        .filter(Favorite.user_id == user_id, Favorite.symbol == normalized)
        .first()
    )
    if existing:
        return existing

    favorite = Favorite(user_id=user_id, symbol=normalized)
    db.add(favorite)
    try:
        db.commit()
    except IntegrityError:
        # Race: another request inserted the same (user_id, symbol) first.
        db.rollback()
        existing = (
            db.query(Favorite)
            .filter(Favorite.user_id == user_id, Favorite.symbol == normalized)
            .first()
        )
        if existing:
            return existing
        raise
    db.refresh(favorite)
    return favorite


def remove_favorite(db: Session, user_id: int, symbol: str) -> bool:
    normalized = symbol.strip().upper()
    deleted = (
        db.query(Favorite)
        .filter(Favorite.user_id == user_id, Favorite.symbol == normalized)
        .delete()
    )
    db.commit()
    return deleted > 0
