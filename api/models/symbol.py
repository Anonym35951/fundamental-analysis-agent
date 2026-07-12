from datetime import datetime
from api.utils.time import utcnow

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from api.models.base import Base


class Symbol(Base):
    """Durchsuchbares Universum aller NYSE/NASDAQ-Stammaktien (siehe
    scripts/import_symbols.py). `sector`/`industry` sind optional und werden
    per Skript-Lauf (--enrich) nachtraeglich befuellt - fehlen sie, greift
    zur Analysezeit ein lazy yfinance-Lookup (agent/DataLoader.get_company_profile),
    unabhaengig von dieser Tabelle."""

    __tablename__ = "symbols"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    exchange: Mapped[str] = mapped_column(String(20), nullable=False)

    sector: Mapped[str | None] = mapped_column(String(100), nullable=True)
    industry: Mapped[str | None] = mapped_column(String(150), nullable=True)

    # False, wenn beim letzten Import-Lauf nicht mehr in der Boersen-Liste
    # enthalten (delisted) - Zeile bleibt fuer Historie/Referenzen erhalten,
    # taucht aber nicht mehr in der Suche auf.
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # NULL = noch nie angereichert; gesetzt (auch bei Fehlschlag mit
    # sector=None) verhindert endloses Retry bei jedem Skript-Lauf.
    enriched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
