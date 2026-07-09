# api/routes/custom_analysis.py
from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.orm import Session
from typing import List
import logging

from api.core.rate_limit import limiter

from api.services.job_manager import (
    job_manager,
    GENERIC_JOB_ERROR,
    TOO_MANY_ACTIVE_JOBS_DETAIL,
)
from api.services.metric_catalog import get_catalog_entries, call_metric
from api.services.custom_analysis_limit import check_can_save_definition
from api.services.event_service import log_event
from api.utils.json_sanitize import make_json_safe
from api.core.dependencies import (
    get_current_user,
    require_analysis_access,
    require_analysis_access_for_units,
    get_db,
)
from api.core.database import SessionLocal
from api.models.user import User
from api.models.analysis_history import AnalysisHistory
from api.crud.analysis_history import create_history_entry, update_history_status
from api.crud import custom_analysis_definition as definitions_crud
from api.schemas.custom_analysis_definition import (
    MetricSelection,
    CustomAnalysisDefinitionCreate,
    CustomAnalysisDefinitionUpdate,
    CustomAnalysisDefinitionRead,
    CustomAnalysisDefinitionRunRequest,
)
from api.routes.analyze import get_action

router = APIRouter(prefix="/analyze/custom", tags=["custom-analysis"])

logger = logging.getLogger(__name__)

_GENERIC_METRIC_ERROR = (
    "Kennzahl konnte nicht berechnet werden — Datenquelle vorübergehend "
    "nicht verfügbar oder Daten unvollständig."
)


def _norm_symbol(symbol: str) -> str:
    return symbol.strip().upper()


@router.get("/metrics")
def list_custom_metrics():
    return get_catalog_entries()


@router.get("/history")
@limiter.limit("30/minute")
def get_metric_history(
    request: Request,
    key: str,
    symbol: str,
    frequency: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    current_user: User = Depends(require_analysis_access),
):
    """Synchronous single-metric lookup for an arbitrary symbol, used by the
    chart-layer builder to overlay metrics/symbols that aren't part of the
    current analysis job. Reuses the same dispatch/normalization the
    multi-metric custom-analysis job uses, just without spawning a job.

    Gated like every other computation endpoint (require_analysis_access:
    verified email + Free-tier monthly quota) plus a rate limit — this used
    to only depend on get_current_user, making it the one uncapped,
    unverified compute path left after the metric_routes.py quota fix."""
    if key not in get_catalog_entries_keys():
        raise HTTPException(status_code=400, detail=f"Unbekannte Metrik: {key}")

    action = get_action()
    symbol = _norm_symbol(symbol)

    params: dict = {}
    if frequency:
        params["frequency"] = frequency
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    try:
        raw_result = call_metric(action, key, symbol, params)
        result = _wrap_metric_result(raw_result, None)
    except Exception:
        logger.exception("Metric history failed: key=%s symbol=%s", key, symbol)
        result = {"value": None, "error": _GENERIC_METRIC_ERROR}

    return {"key": key, "symbol": symbol, **result}


def _evaluate_criterion(operator: str, threshold: float, value) -> bool | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if operator == ">":
        return value > threshold
    if operator == "<":
        return value < threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<=":
        return value <= threshold
    return None


def _wrap_metric_result(raw, criterion: dict | None) -> dict:
    # Historical metrics return a pandas DataFrame; flatten to the generic
    # {date, value}[] series shape the frontend's Sparkline/TimeSeriesChart
    # components expect, picking the first non-index column as "value".
    if hasattr(raw, "to_dict") and hasattr(raw, "columns"):
        value_column = raw.columns[0] if len(raw.columns) > 0 else None
        series = [
            {"date": str(idx), "value": row[value_column]}
            for idx, row in raw.iterrows()
        ] if value_column is not None else []
        return {"value": None, "series": make_json_safe(series)}

    safe = make_json_safe(raw)

    if isinstance(safe, dict) and "error" in safe:
        return {"value": None, "error": str(safe["error"])}

    result = {"value": safe}

    if criterion is not None:
        meets = _evaluate_criterion(criterion["operator"], criterion["threshold"], safe)
        if meets is not None:
            result["meets_criterion"] = meets

    return result


def _launch_custom_job(
    db: Session,
    user: User,
    symbol: str,
    frequency: str | None,
    metrics: List[MetricSelection],
    definition_id: int | None,
) -> str:
    if (
        job_manager.count_active_jobs(user.id)
        >= job_manager.max_active_jobs_per_user
    ):
        raise HTTPException(status_code=429, detail=TOO_MANY_ACTIVE_JOBS_DETAIL)

    job_id = job_manager.create_job(symbol=symbol, total=len(metrics), user_id=user.id)
    create_history_entry(db, user.id, job_id, symbol, "custom", frequency)
    if definition_id is not None:
        db.query(AnalysisHistory).filter_by(job_id=job_id).update({"definition_id": definition_id})
        db.commit()
    log_event(
        db,
        "analysis_started",
        user_id=user.id,
        metadata={
            "mode": "custom",
            "frequency": frequency,
            "symbol": symbol,
            "metric_count": len(metrics),
        },
    )

    def run():
        history_db = SessionLocal()
        try:
            action = get_action()
            job_manager.set_current(job_id, "Starte eigene Analyse…")

            for selection in metrics:
                params = dict(selection.params or {})
                if frequency and "frequency" not in params:
                    params["frequency"] = frequency

                job_manager.set_current(job_id, selection.key)

                try:
                    raw_result = call_metric(action, selection.key, symbol, params)
                    criterion = selection.criterion.model_dump() if selection.criterion else None
                    metric_result = _wrap_metric_result(raw_result, criterion)
                except Exception:
                    logger.exception(
                        "Custom metric failed: job_id=%s key=%s symbol=%s",
                        job_id,
                        selection.key,
                        symbol,
                    )
                    metric_result = {"value": None, "error": _GENERIC_METRIC_ERROR}

                job_manager.add_result(job_id, selection.key, metric_result)

            job_manager.set_done(job_id)
            snapshot = {
                "metrics": job_manager.get_result(job_id)["results"],
                "selection": [m.model_dump() for m in metrics],
            }
            update_history_status(history_db, job_id, "done", snapshot)
        except Exception:
            logger.exception(
                "Custom analysis job failed: job_id=%s symbol=%s", job_id, symbol
            )
            job_manager.set_error(job_id, GENERIC_JOB_ERROR)
            update_history_status(history_db, job_id, "error")
        finally:
            history_db.close()

    job_manager.submit(run)
    return job_id


class CustomAnalysisStartRequest(CustomAnalysisDefinitionRunRequest):
    metrics: List[MetricSelection]


@router.post("/start")
@limiter.limit("30/minute")
def start_custom_analysis(
    request: Request,
    payload: CustomAnalysisStartRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not payload.metrics:
        raise HTTPException(status_code=400, detail="Bitte mindestens eine Metrik auswählen.")

    unknown = [m.key for m in payload.metrics if m.key not in get_catalog_entries_keys()]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unbekannte Metriken: {', '.join(unknown)}")

    require_analysis_access_for_units(db, current_user, units=len(payload.metrics))

    symbol = _norm_symbol(payload.symbol)
    job_id = _launch_custom_job(db, current_user, symbol, payload.frequency, payload.metrics, None)
    return {"job_id": job_id, "symbol": symbol}


def get_catalog_entries_keys() -> set[str]:
    return {entry["key"] for entry in get_catalog_entries()}


@router.get("/{job_id}/progress")
def get_custom_progress(job_id: str, current_user: User = Depends(get_current_user)):
    p = job_manager.get_progress(job_id, user_id=current_user.id)
    if not p:
        raise HTTPException(status_code=404, detail="Job not found")
    return p


@router.get("/{job_id}/result")
def get_custom_result(job_id: str, current_user: User = Depends(get_current_user)):
    r = job_manager.get_result(job_id, user_id=current_user.id)
    if not r:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": r["job_id"],
        "symbol": r["symbol"],
        "status": r["status"],
        "metrics": r["results"],
    }


# =============================
# SAVED DEFINITIONS (CRUD + run)
# =============================

@router.post("/definitions", response_model=CustomAnalysisDefinitionRead)
def create_definition(
    payload: CustomAnalysisDefinitionCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    unknown = [m.key for m in payload.metrics if m.key not in get_catalog_entries_keys()]
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unbekannte Metriken: {', '.join(unknown)}")

    check_can_save_definition(db, current_user)

    metrics_json = [m.model_dump() for m in payload.metrics]
    definition = definitions_crud.create_definition(db, current_user.id, payload.name, metrics_json)
    return definition


@router.get("/definitions", response_model=List[CustomAnalysisDefinitionRead])
def list_definitions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return definitions_crud.list_definitions_for_user(db, current_user.id)


@router.get("/definitions/{definition_id}", response_model=CustomAnalysisDefinitionRead)
def get_definition(
    definition_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    definition = definitions_crud.get_definition(db, current_user.id, definition_id)
    if not definition:
        raise HTTPException(status_code=404, detail="Definition not found")
    return definition


@router.patch("/definitions/{definition_id}", response_model=CustomAnalysisDefinitionRead)
def update_definition(
    definition_id: int,
    payload: CustomAnalysisDefinitionUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    definition = definitions_crud.get_definition(db, current_user.id, definition_id)
    if not definition:
        raise HTTPException(status_code=404, detail="Definition not found")

    metrics_json = None
    if payload.metrics is not None:
        unknown = [m.key for m in payload.metrics if m.key not in get_catalog_entries_keys()]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unbekannte Metriken: {', '.join(unknown)}")
        metrics_json = [m.model_dump() for m in payload.metrics]

    return definitions_crud.update_definition(db, definition, name=payload.name, metrics=metrics_json)


@router.delete("/definitions/{definition_id}", status_code=204)
def delete_definition(
    definition_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    definition = definitions_crud.get_definition(db, current_user.id, definition_id)
    if not definition:
        raise HTTPException(status_code=404, detail="Definition not found")
    definitions_crud.delete_definition(db, definition)


@router.post("/definitions/{definition_id}/run")
@limiter.limit("30/minute")
def run_definition(
    request: Request,
    definition_id: int,
    payload: CustomAnalysisDefinitionRunRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    definition = definitions_crud.get_definition(db, current_user.id, definition_id)
    if not definition:
        raise HTTPException(status_code=404, detail="Definition not found")

    metrics = [MetricSelection(**entry) for entry in definition.metrics]
    require_analysis_access_for_units(db, current_user, units=len(metrics))

    symbol = _norm_symbol(payload.symbol)

    job_id = _launch_custom_job(db, current_user, symbol, payload.frequency, metrics, definition.id)
    definitions_crud.mark_definition_run(db, definition)

    return {"job_id": job_id, "symbol": symbol}
