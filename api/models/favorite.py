from datetime import datetime
from api.utils.time import utcnow

from sqlalchemy import String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class Favorite(Base):
    __tablename__ = "favorites"
    __table_args__ = (
        UniqueConstraint("user_id", "symbol", name="uq_favorites_user_id_symbol"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utcnow,
        nullable=False,
    )
