import re
from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator

USERNAME_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{3,50}$")
MIN_AGE = 16


class UserCreate(BaseModel):
    email: EmailStr
    password: str
    username: str
    first_name: str
    last_name: str
    age: int
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

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int) -> int:
        if value < MIN_AGE:
            raise ValueError(f"Mindestalter ist {MIN_AGE} Jahre.")
        return value

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
    age: int | None = None

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

    @field_validator("age")
    @classmethod
    def validate_age(cls, value: int | None) -> int | None:
        if value is not None and value < MIN_AGE:
            raise ValueError(f"Mindestalter ist {MIN_AGE} Jahre.")
        return value


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
    age: int | None
    onboarding_completed_at: datetime | None

    class Config:
        from_attributes = True
