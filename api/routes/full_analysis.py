# api/routes/full_analysis.py
from __future__ import annotations

import logging

from fastapi import APIRouter, Query, HTTPException, Depends, Request
from sqlalchemy.orm import Session

from api.core.rate_limit import limiter
from api.services.job_manager import (
    job_manager as jobs,
    GENERIC_JOB_ERROR,
    TOO_MANY_ACTIVE_JOBS_DETAIL,
)
from api.core.dependencies import get_current_user, require_analysis_access_for_units, get_db
from api.core.database import SessionLocal
from api.models.user import User
from api.crud.analysis_history import create_history_entry, update_history_status
from api.services.event_service import log_event
from api.routes.analyze import get_action
from api.utils.json_sanitize import make_json_safe
from api.utils.symbol_validation import ensure_known_symbol

router = APIRouter(prefix="/full", tags=["full-analysis"])

logger = logging.getLogger(__name__)


def build_analysis_plan():
    """
    Gleiche Logik wie in deinem AgentOrchestrator, nur API-tauglich:
    wir erzeugen eine Liste aus Tasks (name, frequency, callable).

    Nutzt die geteilte AgentAction-Instanz aus analyze.py::get_action()
    statt einer eigenen (LAUNCH_AUDIT.md P2-6) - vorher hatte full_analysis.py
    eine zweite, unabhängige AgentAction() ab Modul-Import-Zeit.
    """
    action = get_action()
    analysis_map = {
        "Wachstumswerte": {
            "func": action.analyze_wachstumswerte,
            "frequencies": ["annual", "quarterly"],
        },
        "Dividendenwerte": {
            "func": action.analyze_dividend_companies,
            "frequencies": ["annual"],
        },
        "Average Grower": {
            "func": action.analyze_average_grower,
            "frequencies": ["annual"],
        },
        "Typische Zykliker": {
            "func": action.analyze_typical_cyclers,
            "frequencies": ["annual", "quarterly"],
        },
        "Zyklische Turnarounds": {
            "func": action.analyze_cycler_turnarounds,
            "frequencies": ["annual", "quarterly"],
        },
        "Optionality": {
            "func": action.analyze_optionality,
            "frequencies": ["annual"],
        },
        "Asset Play": {
            "func": action.analyze_asset_play,
            "frequencies": ["annual"],
        },
    }

    plan = []
    for analysis_name, cfg in analysis_map.items():
        func = cfg["func"]
        for freq in cfg["frequencies"]:
            # eindeutiger key für results
            key = f"{analysis_name}|{freq}"
            plan.append((key, analysis_name, freq, func))
    return plan


def run_full_analysis_job(job_id: str, symbol: str):
    """
    Läuft in einem Thread. Aktualisiert Fortschritt NACH jeder Analyse.
    """
    history_db = SessionLocal()
    try:
        plan = build_analysis_plan()
        jobs.set_reporting_currency(job_id, resolve_reporting_currency(get_action(), symbol))

        for key, analysis_name, freq, func in plan:
            jobs.set_current(job_id, f"{analysis_name} ({freq})")

            # Einige Methoden haben frequency, andere nicht
            if "frequency" in func.__code__.co_varnames:
                result = func(symbol, frequency=freq, use_cache=True)
            else:
                result = func(symbol)

            jobs.add_result(job_id, key, make_json_safe(result))

        jobs.set_done(job_id)
        snapshot = jobs.get_result(job_id)["results"]
        update_history_status(history_db, job_id, "done", snapshot)

    except Exception:
        logger.exception(
            "Full analysis job failed: job_id=%s symbol=%s", job_id, symbol
        )
        jobs.set_error(job_id, GENERIC_JOB_ERROR)
        update_history_status(history_db, job_id, "error")
    finally:
        history_db.close()


@router.post("/start")
@limiter.limit("10/minute")
def start_full_analysis(
    request: Request,
    symbol: str = Query(..., min_length=1, description="Stock symbol z.B. AAPL"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    symbol = symbol.strip().upper()
    # Symbol-Check VOR dem Quota-Verbrauch (gleiche Begründung wie der
    # Active-Jobs-Check direkt darunter, LAUNCH_AUDIT.md P2-3).
    symbol = ensure_known_symbol(db, symbol)

    # Active-Jobs-Check VOR dem Quota-Verbrauch (LAUNCH_AUDIT.md P2-3) - vorher
    # verbrauchte Depends(require_analysis_access) das Kontingent schon, bevor
    # dieser 429-Check überhaupt lief.
    if jobs.count_active_jobs(current_user.id) >= jobs.max_active_jobs_per_user:
        raise HTTPException(status_code=429, detail=TOO_MANY_ACTIVE_JOBS_DETAIL)

    plan = build_analysis_plan()

    # Quota-Einheiten = tatsächliche Rechenlast (LAUNCH_AUDIT.md P2-11):
    # vorher kostete eine Vollanalyse pauschal 1 Einheit, obwohl sie
    # intern len(plan) (~10) Teil-Analysen berechnet - dieselbe Rechenlast
    # wie eine Custom-Analyse mit 10 Kennzahlen, die aber 10 Einheiten
    # gekostet hätte. Betreiber-Entscheidung 2026-07-12: Vollanalyse an die
    # tatsächliche Last angleichen, nicht Custom pauschalisieren.
    require_analysis_access_for_units(db, current_user, units=len(plan))

    job_id = jobs.create_job(symbol=symbol, total=len(plan), user_id=current_user.id)
    create_history_entry(db, current_user.id, job_id, symbol, "full", None)
    log_event(
        db,
        "analysis_started",
        user_id=current_user.id,
        metadata={"mode": "full", "symbol": symbol, "units": len(plan)},
    )

    jobs.submit(lambda: run_full_analysis_job(job_id, symbol))

    return {"job_id": job_id, "symbol": symbol, "total": len(plan)}


@router.get("/full/{job_id}/progress")
def full_progress(job_id: str, current_user: User = Depends(get_current_user)):
    data = jobs.get_progress(job_id, user_id=current_user.id)
    if not data:
        raise HTTPException(status_code=404, detail="job_id nicht gefunden")
    return data


@router.get("/full/{job_id}/result")
def full_result(job_id: str, current_user: User = Depends(get_current_user)):
    data = jobs.get_result(job_id, user_id=current_user.id)
    if not data:
        raise HTTPException(status_code=404, detail="job_id nicht gefunden")
    return data