from datetime import datetime
from api.utils.time import utcnow

from sqlalchemy import DateTime, ForeignKey, Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class CustomerNote(Base):
    """Interne Admin-Notiz zu einem Kunden (leichtgewichtiges CRM). user_id
    ist die/der Kund:in, ueber die die Notiz geschrieben wurde - bei DSGVO-
    Loeschung des Kontos (delete_account) verschwinden auch die Notizen ueber
    diese Person (CASCADE). admin_author_id ist der Admin, der die Notiz
    verfasst hat - bleibt beim Loeschen des Admin-Accounts als historischer
    Eintrag erhalten, verliert nur die Autor-Zuordnung (SET NULL), analog zum
    Muster bei product_events.user_id."""

    __tablename__ = "customer_notes"
    __table_args__ = (
        Index("ix_customer_notes_user_id_created_at", "user_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    admin_author_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )

    note: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
