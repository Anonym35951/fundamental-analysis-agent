from datetime import datetime
from api.utils.time import utcnow

from sqlalchemy import JSON, String, DateTime, Integer, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class AnalysisHistory(Base):
    __tablename__ = "analysis_history"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    job_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    symbol: Mapped[str] = mapped_column(String(20), nullable=False)

    # "full" oder einer der Einzelmodi (wachstumswerte, dividendenwerte, ...)
    mode: Mapped[str] = mapped_column(String(50), nullable=False)

    frequency: Mapped[str | None] = mapped_column(String(20), nullable=True)

    # running | done | error
    status: Mapped[str] = mapped_column(String(20), default="running", nullable=False)

    # Set when this run was launched from a saved CustomAnalysisDefinition
    # (vs. an ad-hoc custom analysis or one of the built-in modes).
    definition_id: Mapped[int | None] = mapped_column(
        ForeignKey("custom_analysis_definitions.id", ondelete="SET NULL"), nullable=True
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=utcnow,
        nullable=False,
    )

    # Full result payload at the time the job finished (per-mode CategoryResult
    # dict, or {"metrics": ..., "selection": ...} for custom-analysis jobs),
    # so history entries remain viewable/re-runnable after the in-memory
    # JobManager state is gone.
    # with_variant: auf Postgres identisch JSONB, ermöglicht aber create_all
    # in SQLite-basierten API-Tests (JSONB kompiliert dort nicht).
    result_snapshot: Mapped[dict | None] = mapped_column(
        JSONB().with_variant(JSON(), "sqlite"), nullable=True
    )
