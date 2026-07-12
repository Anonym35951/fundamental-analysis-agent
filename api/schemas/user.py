import re
from datetime import date, datetime

from pydantic import BaseModel, EmailStr, field_validator

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,50}$")
MIN_AGE = 16


def calculate_age(birth_date: date, *, today: date | None = None) -> int:
    """Ganze Jahre seit birth_date, korrekt fuer noch nicht erreichte
    Geburtstage im laufenden Jahr (nicht nur Jahresdifferenz)."""
    reference = today or date.today()
    years = reference.year - birth_date.year
    if (reference.month, reference.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years


def _validate_birth_date(value: date) -> date:
    today = date.today()
    if value > today:
        raise ValueError("Geburtsdatum darf nicht in der Zukunft liegen.")
    if calculate_age(value, today=today) < MIN_AGE:
        raise ValueError(f"Mindestalter ist {MIN_AGE} Jahre.")
    return value


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    username: str
    first_name: str
    last_name: str
    birth_date: date
    terms_accepted: bool
    privacy_accepted: bool

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str) -> str:
        if not USERNAME_PATTERN.match(value):
            raise ValueError(
                "Benutzername muss 3-50 Zeichen lang sein und darf nur "
                "Buchstaben, Zahlen, Punkt, Unterstrich oder Bindestrich enthalten."
            )
        return value

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Darf nicht leer sein.")
        return value.strip()

    @field_validator("birth_date")
    @classmethod
    def validate_birth_date(cls, value: date) -> date:
        return _validate_birth_date(value)

    @field_validator("terms_accepted")
    @classmethod
    def validate_terms_accepted(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Nutzungsbedingungen müssen akzeptiert werden.")
        return value

    @field_validator("privacy_accepted")
    @classmethod
    def validate_privacy_accepted(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Datenschutzerklärung muss zur Kenntnis genommen werden.")
        return value


class UserProfileUpdateRequest(BaseModel):
    username: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    # Ersetzt `age` (siehe api/models/user.py) - auch fuer Bestandsnutzer, die
    # ihr Profil jetzt erstmals/erneut nachpflegen, wird ab sofort das
    # Geburtsdatum statt eines statischen Alters erfasst.
    birth_date: date | None = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str | None) -> str | None:
        if value is not None and not USERNAME_PATTERN.match(value):
            raise ValueError(
                "Benutzername muss 3-50 Zeichen lang sein und darf nur "
                "Buchstaben, Zahlen, Punkt, Unterstrich oder Bindestrich enthalten."
            )
        return value

    @field_validator("first_name", "last_name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("Darf nicht leer sein.")
        return value.strip() if value is not None else value

    @field_validator("birth_date")
    @classmethod
    def validate_birth_date(cls, value: date | None) -> date | None:
        if value is None:
            return value
        return _validate_birth_date(value)


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    is_active: bool
    is_superuser: bool
    created_at: datetime

    email_verified: bool

    plan: str

    # active | canceling | past_due | canceled
    billing_status: str

    # month | year | None
    billing_interval: str | None

    # Ende der aktuellen Billing-Periode
    current_period_end: datetime | None

    grace_until: datetime | None

    monthly_request_count: int
    monthly_request_limit: int | None

    stripe_customer_id: str | None
    stripe_subscription_id: str | None
    stripe_price_id: str | None

    username: str | None
    first_name: str | None
    last_name: str | None
    # age: Legacy-Feld, nur bei vor der Umstellung registrierten Konten
    # gesetzt (siehe api/models/user.py). birth_date ist der neue,
    # kanonische Wert - Frontend zeigt age nur an, wenn birth_date fehlt.
    age: int | None
    birth_date: date | None
    onboarding_completed_at: datetime | None

    class Config:
        from_attributes = True
