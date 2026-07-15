# api/routes/admin_symbols.py
"""Admin-Endpoints für das Unternehmens-Universum (EVOLVING.md EV-012):
liefert den Beweis, wie viele NYSE/NASDAQ-Symbole tatsächlich aktiv/
durchsuchbar sind (siehe EV-010: die Produktions-Tabelle enthielt lange nur
den 23-Zeilen-Migrations-Seed), und bietet einen manuellen Import-Trigger,
der ohne Render-Shell-Zugriff auskommt."""
import asyncio

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from api.core.database import SessionLocal
from api.core.dependencies import get_db, require_admin
from api.core.rate_limit import limiter
from api.models.symbol import Symbol
from api.models.user import User
from api.services import symbol_sync_service

router = APIRouter(prefix="/admin/symbols", tags=["admin-symbols"])


def _compute_symbol_stats(db: Session) -> dict:
    """Reine, DB-Session-parametrisierte Funktion (kein FastAPI-Dependency-
    Aufruf) - direkt testbar, Stil wie _compute_subscription_stats in
    admin_stats.py."""
    rows = (
        db.query(Symbol.exchange, Symbol.is_active, func.count(Symbol.id))
        .group_by(Symbol.exchange, Symbol.is_active)
        .all()
    )

    by_exchange: dict[str, dict[str, int]] = {}
    total_active = 0
    total_inactive = 0
    for exchange, is_active, count in rows:
        bucket = by_exchange.setdefault(exchange, {"active": 0, "inactive": 0})
        if is_active:
            bucket["active"] += count
            total_active += count
        else:
            bucket["inactive"] += count
            total_inactive += count

    last_updated = db.query(func.max(Symbol.updated_at)).scalar()

    return {
        "total": total_active + total_inactive,
        "active": total_active,
        "inactive": total_inactive,
        "by_exchange": by_exchange,
        "last_updated_at": last_updated.isoformat() if last_updated else None,
    }


@router.get("/stats")
@limiter.limit("20/minute")
def get_symbol_stats(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    return _compute_symbol_stats(db)


@router.post("/refresh", status_code=202)
@limiter.limit("5/minute")
async def refresh_symbols(
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    """Stößt den Import im Hintergrund an (derselbe Lock wie der Startup-
    Check/wöchentliche Worker aus symbol_sync_service.py) und antwortet
    sofort mit 202, statt auf den Abschluss (Netzwerk-Download + DB-Upsert,
    mehrere Sekunden) zu warten."""
    already_running = symbol_sync_service._import_lock.locked()
    asyncio.create_task(symbol_sync_service._run_import_locked(SessionLocal))
    return {"status": "already_running" if already_running else "started"}
