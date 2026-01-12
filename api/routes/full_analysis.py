# api/routes/full_analysis.py
from __future__ import annotations

import threading
from fastapi import APIRouter, Query, HTTPException
from agent.AgentAction import AgentAction
from api.services.job_manager import JobManager

router = APIRouter(prefix="/full", tags=["full-analysis"])

# global (für den Anfang ok)
action = AgentAction()
jobs = JobManager()


def build_analysis_plan():
    """
    Gleiche Logik wie in deinem AgentOrchestrator, nur API-tauglich:
    wir erzeugen eine Liste aus Tasks (name, frequency, callable).
    """
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
    try:
        plan = build_analysis_plan()

        for key, analysis_name, freq, func in plan:
            jobs.set_current(job_id, f"{analysis_name} ({freq})")

            # Einige Methoden haben frequency, andere nicht
            if "frequency" in func.__code__.co_varnames:
                result = func(symbol, frequency=freq, use_cache=True)
            else:
                result = func(symbol)

            jobs.add_result(job_id, key, result)

        jobs.set_done(job_id)

    except Exception as e:
        jobs.set_error(job_id, str(e))


@router.post("/start")
def start_full_analysis(
    symbol: str = Query(..., min_length=1, description="Stock symbol z.B. AAPL"),
):
    symbol = symbol.strip().upper()

    plan = build_analysis_plan()
    job_id = jobs.create_job(symbol=symbol, total=len(plan))

    t = threading.Thread(target=run_full_analysis_job, args=(job_id, symbol), daemon=True)
    t.start()

    return {"job_id": job_id, "symbol": symbol, "total": len(plan)}


@router.get("/full/{job_id}/progress")
def full_progress(job_id: str):
    data = jobs.get_progress(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job_id nicht gefunden")
    return data


@router.get("/full/{job_id}/result")
def full_result(job_id: str):
    data = jobs.get_result(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="job_id nicht gefunden")
    return data