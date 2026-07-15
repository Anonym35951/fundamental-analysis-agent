# api/routes/analyze.py
from fastapi import APIRouter, Query, HTTPException, Depends, Request
from sqlalchemy import case, func, or_
from sqlalchemy.orm import Session
from typing import Callable, Dict
import inspect
import logging

from api.core.rate_limit import limiter

from api.services.event_service import log_event
from api.services.job_manager import (
    job_manager,
    GENERIC_JOB_ERROR,
    TOO_MANY_ACTIVE_JOBS_DETAIL,
)
from api.utils.json_sanitize import make_json_safe
from api.utils.symbol_validation import ensure_known_symbol
from api.utils.reporting_currency import resolve_reporting_currency
from api.core.dependencies import get_current_user, require_analysis_access, get_db
from api.core.database import SessionLocal
from api.models.symbol import Symbol
from api.models.user import User
from api.crud.analysis_history import (
    create_history_entry,
    update_history_status,
    get_recent_history,
    get_history_entry,
)
from api.schemas.analysis_history import AnalysisHistoryEntry, AnalysisHistorySnapshot
from agent.AgentAction import AgentAction

router = APIRouter(prefix="/analyze", tags=["analyze"])

logger = logging.getLogger(__name__)


# =============================
# LAZY ACTION (🔥 WICHTIG)
# =============================

_action_instance: AgentAction | None = None

# Popular-Fallback fuer die leere Suche (Fokus-Dropdown ohne Eingabe) und
# fuer den Fall, dass scripts/import_symbols.py noch nie gelaufen ist (dann
# ist die symbols-Tabelle leer) - die ehemals fest kuratierten 23 Symbole,
# jetzt nur noch als Ticker-Liste statt eigener Datenquelle.
POPULAR_SYMBOLS = [
    "AAPL", "MO", "GOOGL", "TSLA", "AMD", "PYPL", "NVDA", "NKE", "UNH",
    "XPEV", "OCGN", "UAA", "BABA", "LUMN", "TTWO", "BIDU", "JD", "CRSP",
    "NVO", "NFLX", "BYD", "SAP", "ILMN",
]


@router.get("/symbols")
@limiter.limit("60/minute")
def search_symbols(
    request: Request,
    query: str = Query("", max_length=50),
    limit: int = Query(20, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Durchsucht das NYSE+NASDAQ-Symbol-Universum (siehe
    scripts/import_symbols.py) nach Symbol oder Firmennamen - ersetzt die
    fruehere "alle 23 Symbole auf einmal laden"-Route, da das komplette
    Universum (~6-7k Zeilen) fuers Frontend-Autocomplete zu gross waere.
    `sectors` bleibt aus Kompatibilitaetsgruenden ein Array (Dropdown-Code
    degradiert damit sauber, auch wenn Branche noch nicht angereichert ist).
    """
    clean_query = query.strip()

    if not clean_query:
        rows = (
            db.query(Symbol)
            .filter(Symbol.symbol.in_(POPULAR_SYMBOLS), Symbol.is_active.is_(True))
            .all()
        )
        # Fallback, falls der Import-Lauf noch nie stattfand.
        if not rows:
            return [{"symbol": s, "name": s, "sectors": []} for s in POPULAR_SYMBOLS[:limit]]
        order = {s: i for i, s in enumerate(POPULAR_SYMBOLS)}
        rows.sort(key=lambda r: order.get(r.symbol, len(order)))
        return [_symbol_to_dict(r) for r in rows[:limit]]

    pattern_prefix = f"{clean_query}%"
    pattern_contains = f"%{clean_query}%"

    rank = case(
        (Symbol.symbol.ilike(clean_query), 0),
        (Symbol.symbol.ilike(pattern_prefix), 1),
        (Symbol.name.ilike(pattern_prefix), 2),
        else_=3,
    )

    rows = (
        db.query(Symbol)
        .filter(
            Symbol.is_active.is_(True),
            or_(Symbol.symbol.ilike(pattern_contains), Symbol.name.ilike(pattern_contains)),
        )
        .order_by(rank, func.length(Symbol.symbol), Symbol.symbol)
        .limit(limit)
        .all()
    )

    return [_symbol_to_dict(r) for r in rows]


def _symbol_to_dict(row: Symbol) -> dict:
    return {
        "symbol": row.symbol,
        "name": row.name,
        "sectors": [s for s in (row.industry, row.sector) if s],
    }


def get_action() -> AgentAction:
    global _action_instance
    if _action_instance is None:
        _action_instance = AgentAction()
    return _action_instance


# -----------------------------
# Helpers
# -----------------------------
def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


# =============================
# SYNC ENDPOINTS
# =============================

@router.get("/wachstumswerte")
def wachstumswerte(
    symbol: str,
    frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    return make_json_safe(
        get_action().analyze_wachstumswerte(_norm_symbol(symbol), frequency)
    )


@router.get("/dividendenwerte")
def dividendenwerte(
    symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    return make_json_safe(
        get_action().analyze_dividend_companies(_norm_symbol(symbol))
    )


@router.get("/average-grower")
def average_grower(
    symbol: str,
    current_user: User = Depends(require_analysis_access),
):
    return make_json_safe(
        get_action().analyze_average_grower(_norm_symbol(symbol))
    )


@router.get("/typische-zykliker")
def typische_zykliker(
    symbol: str,
    frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    return make_json_safe(
        get_action().analyze_typical_cyclers(_norm_symbol(symbol), frequency)
    )


@router.get("/turnarounds")
def turnarounds(
    symbol: str,
    frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    return make_json_safe(
        get_action().analyze_cycler_turnarounds(_norm_symbol(symbol), frequency)
    )


@router.get("/optionality")
def optionality(
    symbol: str,
    frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    return make_json_safe(
        get_action().analyze_optionality(_norm_symbol(symbol), frequency)
    )


@router.get("/asset-play")
def asset_play(
    symbol: str,
    frequency: str = "annual",
    current_user: User = Depends(require_analysis_access),
):
    return make_json_safe(
        get_action().analyze_asset_play(_norm_symbol(symbol), frequency)
    )


# =============================
# JOB-BASED SINGLE ANALYSES
# =============================

DISPLAY_NAME: Dict[str, str] = {
    "wachstumswerte": "Wachstumswerte",
    "dividendenwerte": "Dividendenwerte",
    "average-grower": "Average Grower",
    "typische-zykliker": "Typische Zykliker",
    "turnarounds": "Zyklische Turnarounds",
    "optionality": "Optionality",
    "asset-play": "Asset Play",
}


def get_analysis_registry() -> Dict[str, Callable[..., dict]]:
    action = get_action()
    return {
        "wachstumswerte": action.analyze_wachstumswerte,
        "dividendenwerte": action.analyze_dividend_companies,
        "average-grower": action.analyze_average_grower,
        "typische-zykliker": action.analyze_typical_cyclers,
        "turnarounds": action.analyze_cycler_turnarounds,
        "optionality": action.analyze_optionality,
        "asset-play": action.analyze_asset_play,
    }


@router.post("/{mode}/start")
@limiter.limit("30/minute")
def start_single_analysis(
    request: Request,
    mode: str,
    symbol: str = Query(...),
    frequency: str = Query("annual"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    registry = get_analysis_registry()

    if mode not in registry:
        raise HTTPException(status_code=404, detail="Unknown analysis mode")

    symbol = _norm_symbol(symbol)
    # Symbol-Check VOR dem Quota-Verbrauch (gleiche Begründung wie der
    # Active-Jobs-Check direkt darunter, LAUNCH_AUDIT.md P2-3): ein
    # unbekanntes/delistetes Symbol soll keine Analyse-Einheit kosten.
    symbol = ensure_known_symbol(db, symbol)

    # Active-Jobs-Check VOR dem Quota-Verbrauch (LAUNCH_AUDIT.md P2-3) - vorher
    # verbrauchte Depends(require_analysis_access) das Kontingent schon, bevor
    # dieser 429-Check überhaupt lief (ebenso beim unbekannten-Modus-404 oben).
    if (
        job_manager.count_active_jobs(current_user.id)
        >= job_manager.max_active_jobs_per_user
    ):
        raise HTTPException(status_code=429, detail=TOO_MANY_ACTIVE_JOBS_DETAIL)

    require_analysis_access(db, current_user)

    pretty_name = DISPLAY_NAME.get(mode, mode)
    key = f"{pretty_name}|{frequency}"

    job_id = job_manager.create_job(symbol=symbol, total=1, user_id=current_user.id)
    create_history_entry(db, current_user.id, job_id, symbol, mode, frequency)
    log_event(
        db,
        "analysis_started",
        user_id=current_user.id,
        metadata={"mode": mode, "frequency": frequency, "symbol": symbol},
    )

    def run():
        history_db = SessionLocal()
        try:
            job_manager.set_current(job_id, "Starte Analyse…")
            job_manager.set_reporting_currency(job_id, resolve_reporting_currency(get_action(), symbol))

            fn = registry[mode]
            sig = inspect.signature(fn)
            kwargs = {}

            if "frequency" in sig.parameters:
                kwargs["frequency"] = frequency
            if "use_cache" in sig.parameters:
                kwargs["use_cache"] = True

            raw_result = fn(symbol, **kwargs) if kwargs else fn(symbol)
            safe_result = make_json_safe(raw_result)

            if not safe_result:
                safe_result = {
                    "overall_assessment": "Keine Daten",
                    "message": "Analyse lieferte kein verwertbares Ergebnis",
                }

            job_manager.set_current(job_id, "Speichere Ergebnis…")
            job_manager.add_result(job_id, key, safe_result)
            job_manager.set_done(job_id)
            snapshot = job_manager.get_result(job_id)["results"]
            update_history_status(history_db, job_id, "done", snapshot)

        except Exception:
            logger.exception(
                "Single analysis job failed: job_id=%s symbol=%s mode=%s",
                job_id,
                symbol,
                mode,
            )
            job_manager.set_error(job_id, GENERIC_JOB_ERROR)
            update_history_status(history_db, job_id, "error")
        finally:
            history_db.close()

    job_manager.submit(run)

    return {
        "job_id": job_id,
        "symbol": symbol,
        "mode": mode,
        "frequency": frequency,
    }


@router.get("/history", response_model=list[AnalysisHistoryEntry])
def get_analysis_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return get_recent_history(db, current_user.id, limit=10)


@router.get("/history/{history_id}/snapshot", response_model=AnalysisHistorySnapshot)
def get_analysis_history_snapshot(
    history_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    entry = get_history_entry(db, current_user.id, history_id)
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found")
    return entry


@router.get("/{mode}/{job_id}/progress")
def get_single_progress(
    mode: str,
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    p = job_manager.get_progress(job_id, user_id=current_user.id)
    if not p:
        raise HTTPException(status_code=404, detail="Job not found")
    return p


@router.get("/{mode}/{job_id}/result")
def get_single_result(
    mode: str,
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    r = job_manager.get_result(job_id, user_id=current_user.id)
    if not r:
        raise HTTPException(status_code=404, detail="Job not found")
    return r