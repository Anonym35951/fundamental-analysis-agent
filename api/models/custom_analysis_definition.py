from datetime import datetime
from api.utils.time import utcnow

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class CustomAnalysisDefinition(Base):
    """A user-defined, reusable bundle of metric selections (+ optional
    threshold criteria). Intentionally has no `symbol` column — a definition
    is independent of any specific stock; the symbol is supplied only when
    the definition is run (see POST /analyze/custom/definitions/{id}/run).
    """

    __tablename__ = "custom_analysis_definitions"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    name: Mapped[str] = mapped_column(String(255), nullable=False)

    # List[{"key": str, "params": dict, "criterion": {"operator": str, "threshold": float} | None}]
    # Order-preserving JSON list. Heterogeneous per-metric params (frequency,
    # date ranges, min_years, ...) and the fact that this is always read/
    # written as a whole unit per definition make a normalized child table
    # unnecessary overhead.
    metrics: Mapped[list] = mapped_column(JSON, nullable=False)

    last_run_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
