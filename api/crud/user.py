from datetime import date, datetime
from api.utils.time import utcnow

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.models.user import User
from api.core.security import hash_password
from api.schemas.user import UserProfileUpdateRequest


def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()


def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()


def get_user_by_email_or_username(db: Session, identifier: str):
    """Login akzeptiert Email ODER Benutzername im selben Formularfeld.
    Email zuerst pruefen (bereits eindeutig indexiert, haeufigster Fall),
    dann auf Benutzername ausweichen."""
    user = get_user_by_email(db, identifier)
    if user is not None:
        return user
    return get_user_by_username(db, identifier)


def create_user(
    db: Session,
    email: str,
    password: str,
    username: str,
    first_name: str,
    last_name: str,
    birth_date: date,
    terms_version: str,
    privacy_version: str,
):
    now = utcnow()
    user = User(
        email=email,
        hashed_password=hash_password(password),
        username=username,
        first_name=first_name,
        last_name=last_name,
        birth_date=birth_date,
        terms_accepted_at=now,
        terms_version=terms_version,
        privacy_accepted_at=now,
        privacy_version=privacy_version,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def update_user_profile(db: Session, user: User, data: UserProfileUpdateRequest) -> User:
    """Partial Update fuer die optionale Profil-Nachpflege (Bestandsnutzer)
    und kuenftige Profil-Bearbeitung. Wendet nur gesetzte Felder an."""
    updates = data.model_dump(exclude_unset=True, exclude_none=True)
    for field, value in updates.items():
        setattr(user, field, value)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise ValueError("Benutzername bereits vergeben")

    db.refresh(user)
    return user
