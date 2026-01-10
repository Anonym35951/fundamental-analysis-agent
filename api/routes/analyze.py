# api/routes/analyze.py
from fastapi import APIRouter, Query, HTTPException
from typing import Callable, Dict
import inspect
import traceback

from agent.ActionModule import AgentAction
from api.services.job_manager import job_manager
from api.utils.json_sanitize import make_json_safe

router = APIRouter(prefix="/analyze", tags=["analyze"])

action = AgentAction()

# -----------------------------
# Helpers
# -----------------------------
def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


# =============================
# SYNC ENDPOINTS (BLEIBEN)
# =============================

@router.get("/wachstumswerte")
def wachstumswerte(symbol: str, frequency: str = "annual"):
    return make_json_safe(action.analyze_wachstumswerte(_norm_symbol(symbol), frequency))


@router.get("/dividendenwerte")
def dividendenwerte(symbol: str):
    return make_json_safe(action.analyze_dividend_companies(_norm_symbol(symbol)))


@router.get("/average-grower")
def average_grower(symbol: str):
    return make_json_safe(action.analyze_average_grower(_norm_symbol(symbol)))


@router.get("/typische-zykliker")
def typische_zykliker(symbol: str, frequency: str = "annual"):
    return make_json_safe(action.analyze_typical_cyclers(_norm_symbol(symbol), frequency))


@router.get("/turnarounds")
def turnarounds(symbol: str, frequency: str = "annual"):
    return make_json_safe(action.analyze_cycler_turnarounds(_norm_symbol(symbol), frequency))


@router.get("/optionality")
def optionality(symbol: str, frequency: str = "annual"):
    return make_json_safe(action.analyze_optionality(_norm_symbol(symbol), frequency))


@router.get("/asset-play")
def asset_play(symbol: str, frequency: str = "annual"):
    return make_json_safe(action.analyze_asset_play(_norm_symbol(symbol), frequency))


# =============================
# JOB-BASED SINGLE ANALYSES ✅
# =============================

ANALYSIS_REGISTRY: Dict[str, Callable[..., dict]] = {
    "wachstumswerte": action.analyze_wachstumswerte,
    "dividendenwerte": action.analyze_dividend_companies,
    "average-grower": action.analyze_average_grower,
    "typische-zykliker": action.analyze_typical_cyclers,
    "turnarounds": action.analyze_cycler_turnarounds,
    "optionality": action.analyze_optionality,
    "asset-play": action.analyze_asset_play,
}

# ✅ Damit die Keys identisch zur Full Analysis aussehen:
DISPLAY_NAME: Dict[str, str] = {
    "wachstumswerte": "Wachstumswerte",
    "dividendenwerte": "Dividendenwerte",
    "average-grower": "Average Grower",
    "typische-zykliker": "Typische Zykliker",
    "turnarounds": "Zyklische Turnarounds",
    "optionality": "Optionality",
    "asset-play": "Asset Play",
}


@router.post("/{mode}/start")
def start_single_analysis(
    mode: str,
    symbol: str = Query(...),
    frequency: str = Query("annual"),
):
    if mode not in ANALYSIS_REGISTRY:
        raise HTTPException(status_code=404, detail="Unknown analysis mode")

    symbol = _norm_symbol(symbol)

    pretty_name = DISPLAY_NAME.get(mode, mode)
    key = f"{pretty_name}|{frequency}"

    job_id = job_manager.create_job(symbol=symbol, total=1)

    def run():
        try:
            job_manager.set_current(job_id, "Starte Analyse…")

            fn = ANALYSIS_REGISTRY[mode]

            # ✅ NICHT TypeError catchen (kann aus der Analyse selbst kommen)
            # Stattdessen prüfen wir sauber, welche Parameter akzeptiert werden.
            sig = inspect.signature(fn)
            params = sig.parameters

            kwargs = {}
            if "frequency" in params:
                kwargs["frequency"] = frequency
            if "use_cache" in params:
                kwargs["use_cache"] = True

            raw_result = fn(symbol, **kwargs) if kwargs else fn(symbol)

            # ✅ WICHTIG: ZUERST sanitizen (hier passieren oft {}-Ergebnisse)
            safe_result = make_json_safe(raw_result)

            # 🛡️ SAFETY: leere / None Ergebnisse NACH Sanitizing abfangen
            if safe_result is None:
                safe_result = {
                    "overall_assessment": "Keine Daten",
                    "message": "Analyse lieferte kein Ergebnis",
                }

            if isinstance(safe_result, dict) and not safe_result:
                safe_result = {
                    "overall_assessment": "Keine Daten",
                    "message": "Analyse lieferte ein leeres Ergebnis (nach JSON-Sanitize)",
                }

            job_manager.set_current(job_id, "Speichere Ergebnis…")
            job_manager.add_result(job_id, key, safe_result)

            job_manager.set_done(job_id)

        except Exception as e:
            # Debug im Backend-Log behalten
            traceback.print_exc()
            job_manager.set_error(job_id, str(e))

    import threading
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