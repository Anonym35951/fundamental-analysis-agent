from datetime import datetime
from api.utils.time import utcnow

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class ProductEvent(Base):
    """Append-only log of product-usage events, für das private Admin-
    Analytics-Dashboard (kein Dritt-Anbieter-Tracking). user_id ist nullable,
    da manche Events vor einem eingeloggten Nutzer entstehen können
    (z. B. anonyme Registrierungsversuche); Löschung des Nutzers (DSGVO)
    setzt die Referenz auf NULL statt die Event-Historie zu löschen, damit
    aggregierte Statistiken (Funnel, MAU-Verläufe) über die Zeit stabil
    bleiben.
    """

    __tablename__ = "product_events"
    __table_args__ = (
        Index("ix_product_events_type_created_at", "event_type", "created_at"),
        Index("ix_product_events_user_id_created_at", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    event_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    # Freiform-Zusatzdaten je Event (z. B. {"mode": "wachstumswerte", "symbol": "AAPL"}
    # oder {"reason": "zu teuer"} für Kündigungsgründe). Kein fester Query-Bedarf
    # auf einzelne Felder -> JSON statt normalisierter Spalten.
    event_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False, index=True
    )
