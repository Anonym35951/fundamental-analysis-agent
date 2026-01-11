# api/routes/analyze.py
from fastapi import APIRouter, Query, HTTPException
from typing import Callable, Dict
import inspect
import traceback
import threading

from api.services.job_manager import job_manager
from api.utils.json_sanitize import make_json_safe
from agent.ActionModule import AgentAction

router = APIRouter(prefix="/analyze", tags=["analyze"])


# =============================
# LAZY ACTION (🔥 WICHTIG)
# =============================

_action_instance: AgentAction | None = None


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
def wachstumswerte(symbol: str, frequency: str = "annual"):
    return make_json_safe(
        get_action().analyze_wachstumswerte(_norm_symbol(symbol), frequency)
    )


@router.get("/dividendenwerte")
def dividendenwerte(symbol: str):
    return make_json_safe(
        get_action().analyze_dividend_companies(_norm_symbol(symbol))
    )


@router.get("/average-grower")
def average_grower(symbol: str):
    return make_json_safe(
        get_action().analyze_average_grower(_norm_symbol(symbol))
    )


@router.get("/typische-zykliker")
def typische_zykliker(symbol: str, frequency: str = "annual"):
    return make_json_safe(
        get_action().analyze_typical_cyclers(_norm_symbol(symbol), frequency)
    )


@router.get("/turnarounds")
def turnarounds(symbol: str, frequency: str = "annual"):
    return make_json_safe(
        get_action().analyze_cycler_turnarounds(_norm_symbol(symbol), frequency)
    )


@router.get("/optionality")
def optionality(symbol: str, frequency: str = "annual"):
    return make_json_safe(
        get_action().analyze_optionality(_norm_symbol(symbol), frequency)
    )


@router.get("/asset-play")
def asset_play(symbol: str, frequency: str = "annual"):
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
def start_single_analysis(
    mode: str,
    symbol: str = Query(...),
    frequency: str = Query("annual"),
):
    registry = get_analysis_registry()

    if mode not in registry:
        raise HTTPException(status_code=404, detail="Unknown analysis mode")

    symbol = _norm_symbol(symbol)
    pretty_name = DISPLAY_NAME.get(mode, mode)
    key = f"{pretty_name}|{frequency}"

    job_id = job_manager.create_job(symbol=symbol, total=1)

    def run():
        try:
            job_manager.set_current(job_id, "Starte Analyse…")

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

        except Exception as e:
            traceback.print_exc()
            job_manager.set_error(job_id, str(e))

    threading.Thread(target=run, daemon=True).start()

    return {
        "job_id": job_id,
        "symbol": symbol,
        "mode": mode,
        "frequency": frequency,
    }


@router.get("/{mode}/{job_id}/progress")
def get_single_progress(mode: str, job_id: str):
    p = job_manager.get_progress(job_id)
    if not p:
        raise HTTPException(status_code=404, detail="Job not found")
    return p


@router.get("/{mode}/{job_id}/result")
def get_single_result(mode: str, job_id: str):
    r = job_manager.get_result(job_id)
    if not r:
        raise HTTPException(status_code=404, detail="Job not found")
    return r