from datetime import date, datetime
from api.utils.time import utcnow

from sqlalchemy import Date, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class FilingAlertState(Base):
    """Zuletzt gesehene 10-K/10-Q-Meldung je Symbol — Grundlage für die
    Filing-Alerts auf Favoriten (api/services/filing_alert_service.py).

    Bewusst pro Symbol statt pro (user_id, symbol) gepflegt: der SEC-Zustand
    ist für alle Nutzer identisch, die dasselbe Symbol favorisiert haben —
    ein gemeinsamer Zustand vermeidet redundante SEC-Abfragen und doppelte
    Update-Logik pro Nutzer.
    """

    __tablename__ = "filing_alert_state"

    symbol: Mapped[str] = mapped_column(String(20), primary_key=True)

    last_seen_accession_number: Mapped[str | None] = mapped_column(
        String(32), nullable=True
    )

    last_seen_form: Mapped[str | None] = mapped_column(String(20), nullable=True)

    last_seen_filing_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    last_checked_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
