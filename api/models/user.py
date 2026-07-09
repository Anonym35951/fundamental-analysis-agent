from datetime import date, datetime
from sqlalchemy import String, Boolean, DateTime, Date, Integer
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
        nullable=False
    )

    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        nullable=False
    )

    # ===== Rollen / Pläne =====
    # free | friends | pro | admin
    plan: Mapped[str] = mapped_column(
        String(50),
        default="free",
        nullable=False
    )

    # ===== Billing-Status =====
    # active | canceling | past_due | canceled
    billing_status: Mapped[str] = mapped_column(
        String(50),
        default="active",
        nullable=False
    )

    # month | year | None
    billing_interval: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )

    # Aktive Stripe-Price-ID des aktuellen Abos
    stripe_price_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    # Ende der aktuellen Billing-Periode
    current_period_end: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    # Grace Period bis Downgrade (z. B. +24h nach Payment Failure)
    grace_until: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    # ===== Usage / Limits =====
    monthly_request_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False
    )

    # Nur für free relevant → sonst None
    monthly_request_limit: Mapped[int | None] = mapped_column(
        Integer,
        default=50,
        nullable=True
    )

    # Erster Tag des Kalendermonats, für den monthly_request_count zählt.
    # Wird bei jedem Quota-Check (try_consume_monthly_request_quota) auf den
    # aktuellen Monat geprüft ("on-read"-Reset) statt per zeitgesteuertem
    # Worker zurückgesetzt zu werden — der Worker konnte am Monatsersten
    # ausfallen, wenn die App gerade deployed/schlafend war (Render).
    # NULL (z. B. bestehende Free-User vor dieser Migration) gilt als "veraltet"
    # und wird beim nächsten Quota-Check normal initialisiert.
    usage_period_start: Mapped[date | None] = mapped_column(
        Date,
        nullable=True
    )

    # ===== Stripe =====
    stripe_customer_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    stripe_subscription_id: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    # ===== Passwort-Reset =====
    # SHA-256-Hash des Reset-Tokens (roher Token geht nur per Mail-Link raus)
    password_reset_token_hash: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    password_reset_expires: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    # ===== E-Mail-Verifizierung (Double-Opt-In) =====
    email_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False
    )

    # Gleiches Muster wie Passwort-Reset: nur der SHA-256-Hash liegt in der DB
    email_verification_token_hash: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True
    )

    email_verification_expires: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    # ===== Profil (optional, auch für Bestandsnutzer nachpflegbar) =====
    username: Mapped[str | None] = mapped_column(
        String(50),
        unique=True,
        index=True,
        nullable=True
    )

    first_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    last_name: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True
    )

    # Nur die Zahl, kein Geburtsdatum (Datenminimierung) — selbstzertifiziert
    # bei der Registrierung, keine Altersverifikation.
    age: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True
    )

    # ===== Rechts-Zustimmung (granular: ToS und Datenschutz getrennt) =====
    terms_accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    terms_version: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )

    privacy_accepted_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )

    privacy_version: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True
    )

    # Zeitpunkt, zu dem der Nutzer das gefuehrte Onboarding (react-joyride-Tour)
    # abgeschlossen oder uebersprungen hat. Fuer Bestandsnutzer bei der
    # Migration auf created_at gesetzt, damit die Tour ihnen nicht nachtraeglich
    # aufgezwungen wird.
    onboarding_completed_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True
    )