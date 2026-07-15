"""Regressionstests für LAUNCH_AUDIT.md P2-3: der Active-Jobs-429-Check muss
VOR dem Quota-Verbrauch laufen, nicht danach - sonst verliert ein Free-User
eine Analyse-Einheit, obwohl der Job mangels freiem Slot nie gestartet wird.

Stil wie test_account_deletion.py/test_audit_log_p2_21.py: direkte
Funktionsaufrufe gegen die In-Memory-SQLite-db-Fixture, kein TestClient.
Der Active-Jobs-Check wird über ein Monkeypatch von
job_manager.count_active_jobs erzwungen; require_analysis_access(_for_units)
wird durch eine Sentinel-Funktion ersetzt, die bei Aufruf fehlschlägt - so
ist bewiesen, dass die Quota-Funktion beim 429 gar nicht erst erreicht wird.
"""
import pytest
from fastapi import HTTPException
from starlette.requests import Request

from api.models.base import Base
from api.models.custom_analysis_definition import CustomAnalysisDefinition
from api.models.symbol import Symbol
from api.models.user import User
import api.routes.analyze as analyze_module
import api.routes.custom_analysis as custom_analysis_module
import api.routes.full_analysis as full_analysis_module
from api.routes.analyze import start_single_analysis
from api.routes.custom_analysis import (
    CustomAnalysisStartRequest,
    run_definition,
    start_custom_analysis,
)
from api.routes.full_analysis import start_full_analysis
from api.schemas.custom_analysis_definition import CustomAnalysisDefinitionRunRequest


@pytest.fixture
def db_full(db):
    # Symbol.__table__ seit EVOLVING.md EV-014 noetig: alle vier Start-
    # Routen pruefen das Symbol jetzt per ensure_known_symbol gegen die
    # symbols-Tabelle, bevor der 429-Check bzw. Quota-Verbrauch greift.
    Base.metadata.create_all(
        db.get_bind(), tables=[CustomAnalysisDefinition.__table__, Symbol.__table__]
    )
    return db


def _make_user(db, **kwargs) -> User:
    defaults = {"email": f"user{id(kwargs)}@example.com", "hashed_password": "x", "plan": "free"}
    defaults.update(kwargs)
    user = User(**defaults)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def _forbid_quota_call(name):
    def _fail(*args, **kwargs):
        raise AssertionError(f"{name} darf beim 429 nicht aufgerufen werden")
    return _fail


def test_single_analysis_429_before_quota(db_full, monkeypatch):
    db = db_full
    user = _make_user(db)
    monkeypatch.setattr(
        analyze_module.job_manager, "count_active_jobs", lambda user_id: 999
    )
    monkeypatch.setattr(
        analyze_module, "require_analysis_access", _forbid_quota_call("require_analysis_access")
    )

    with pytest.raises(HTTPException) as exc:
        start_single_analysis(
            request=_fake_request(),
            mode="wachstumswerte",
            symbol="AAPL",
            frequency="annual",
            current_user=user,
            db=db,
        )
    assert exc.value.status_code == 429


def test_full_analysis_429_before_quota(db_full, monkeypatch):
    db = db_full
    user = _make_user(db)
    monkeypatch.setattr(
        full_analysis_module.jobs, "count_active_jobs", lambda user_id: 999
    )
    monkeypatch.setattr(
        full_analysis_module,
        "require_analysis_access_for_units",
        _forbid_quota_call("require_analysis_access_for_units"),
    )

    with pytest.raises(HTTPException) as exc:
        start_full_analysis(request=_fake_request(), symbol="AAPL", current_user=user, db=db)
    assert exc.value.status_code == 429


def test_custom_analysis_start_429_before_quota(db_full, monkeypatch):
    db = db_full
    user = _make_user(db)
    monkeypatch.setattr(
        custom_analysis_module.job_manager, "count_active_jobs", lambda user_id: 999
    )
    monkeypatch.setattr(
        custom_analysis_module,
        "require_analysis_access_for_units",
        _forbid_quota_call("require_analysis_access_for_units"),
    )
    payload = CustomAnalysisStartRequest(symbol="AAPL", metrics=[{"key": "calculate_KGV", "params": {}}])

    with pytest.raises(HTTPException) as exc:
        start_custom_analysis(request=_fake_request(), payload=payload, current_user=user, db=db)
    assert exc.value.status_code == 429


def test_run_definition_429_before_quota(db_full, monkeypatch):
    from api.crud import custom_analysis_definition as definitions_crud

    db = db_full
    user = _make_user(db)
    definition = definitions_crud.create_definition(
        db, user.id, "Meine Analyse", [{"key": "calculate_KGV", "params": {}}]
    )
    monkeypatch.setattr(
        custom_analysis_module.job_manager, "count_active_jobs", lambda user_id: 999
    )
    monkeypatch.setattr(
        custom_analysis_module,
        "require_analysis_access_for_units",
        _forbid_quota_call("require_analysis_access_for_units"),
    )
    payload = CustomAnalysisDefinitionRunRequest(symbol="AAPL")

    with pytest.raises(HTTPException) as exc:
        run_definition(
            request=_fake_request(),
            definition_id=definition.id,
            payload=payload,
            current_user=user,
            db=db,
        )
    assert exc.value.status_code == 429
