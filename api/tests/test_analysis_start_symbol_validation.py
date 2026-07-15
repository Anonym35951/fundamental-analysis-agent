"""Regressionstests für EVOLVING.md EV-014: alle vier Analyse-Start-
Einstiegspunkte (start_single_analysis, start_full_analysis,
start_custom_analysis, run_definition) müssen ein unbekanntes/delistetes
Symbol früh mit 422 ablehnen, statt es unbeaufsichtigt an den Agenten
durchzureichen - und dürfen dabei keine heute gültige Analyse blockieren
(inkl. Punkt-Schreibweise wie "brk.b").

job_manager.submit wird überall gemockt (Stil wie
test_full_analysis_quota_units_p2_11.py), damit kein echter Analyse-Job im
Thread-Pool läuft - dieser Test prüft nur, ob der Start selbst (400/422 vs.
200/202) korrekt entschieden wird.
"""
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from starlette.requests import Request

import api.routes.analyze as analyze_module
import api.routes.custom_analysis as custom_analysis_module
import api.routes.full_analysis as full_analysis_module
from api.models.analysis_history import AnalysisHistory
from api.models.base import Base
from api.models.custom_analysis_definition import CustomAnalysisDefinition
from api.models.symbol import Symbol
from api.models.user import User
from api.routes.analyze import start_single_analysis
from api.routes.custom_analysis import (
    CustomAnalysisStartRequest,
    run_definition,
    start_custom_analysis,
)
from api.routes.full_analysis import start_full_analysis
from api.schemas.custom_analysis_definition import CustomAnalysisDefinitionRunRequest


@pytest.fixture
def symbols_db(db):
    Base.metadata.create_all(
        db.get_bind(),
        tables=[
            Symbol.__table__,
            AnalysisHistory.__table__,
            CustomAnalysisDefinition.__table__,
        ],
    )
    db.add_all(
        [
            Symbol(symbol="AAPL", name="Apple Inc.", exchange="NASDAQ", is_active=True),
            Symbol(symbol="BRK-B", name="Berkshire Hathaway Inc", exchange="NYSE", is_active=True),
            Symbol(symbol="DEADCO", name="Delisted Co", exchange="NASDAQ", is_active=False),
        ]
    )
    db.commit()
    return db


def _make_user(db, **kwargs) -> User:
    defaults = {
        "email": f"user{id(kwargs)}@example.com",
        "hashed_password": "x",
        "plan": "free",
        "email_verified": True,
        "monthly_request_limit": 50,
        "monthly_request_count": 0,
    }
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


# ---------------------------------------------------------------------------
# start_single_analysis (api/routes/analyze.py)
# ---------------------------------------------------------------------------

def _isolate_job_manager(manager, monkeypatch):
    """`job_manager`/`jobs` ist ein prozessweiter Singleton (dieselbe Instanz
    unter allen drei Routen-Modulen), der zwischen Tests NICHT zurueckgesetzt
    wird. Da jede frische In-Memory-SQLite-DB Autoincrement-User-IDs wieder
    bei 1 beginnen laesst, wuerde ein per create_job() echt angelegter,
    dauerhaft "running" bleibender Job (submit ist gemockt, set_done wird
    dadurch nie aufgerufen) spaetere Tests mit derselben user_id verfaelschen
    (z. B. den 429-Active-Jobs-Check in test_full_analysis_quota_units_p2_11).
    submit/create_job werden daher fuer die "accepts"-Tests dieser Datei
    vollstaendig isoliert."""
    monkeypatch.setattr(manager, "submit", MagicMock())
    monkeypatch.setattr(manager, "count_active_jobs", lambda user_id: 0)
    monkeypatch.setattr(manager, "create_job", MagicMock(return_value="test-job-id"))


def test_start_single_analysis_accepts_known_active_symbol(symbols_db, monkeypatch):
    _isolate_job_manager(analyze_module.job_manager, monkeypatch)
    user = _make_user(symbols_db)

    result = start_single_analysis(
        request=_fake_request(), mode="wachstumswerte", symbol="AAPL",
        frequency="annual", current_user=user, db=symbols_db,
    )

    assert result["symbol"] == "AAPL"


def test_start_single_analysis_rejects_unknown_symbol_with_422(symbols_db, monkeypatch):
    monkeypatch.setattr(analyze_module.job_manager, "submit", MagicMock())
    user = _make_user(symbols_db)

    with pytest.raises(HTTPException) as exc:
        start_single_analysis(
            request=_fake_request(), mode="wachstumswerte", symbol="FOOBARX",
            frequency="annual", current_user=user, db=symbols_db,
        )
    assert exc.value.status_code == 422
    assert "nicht im unterstützten" in exc.value.detail


def test_start_single_analysis_rejects_delisted_symbol_with_distinct_message(symbols_db, monkeypatch):
    monkeypatch.setattr(analyze_module.job_manager, "submit", MagicMock())
    user = _make_user(symbols_db)

    with pytest.raises(HTTPException) as exc:
        start_single_analysis(
            request=_fake_request(), mode="wachstumswerte", symbol="DEADCO",
            frequency="annual", current_user=user, db=symbols_db,
        )
    assert exc.value.status_code == 422
    assert "nicht mehr gelistet" in exc.value.detail


def test_start_single_analysis_accepts_lowercase_dot_style_share_class(symbols_db, monkeypatch):
    """"brk.b" (Kleinschreibung + Punkt, wie ein Nutzer es eintippen könnte)
    muss auf die als "BRK-B" gespeicherte Zeile treffen."""
    _isolate_job_manager(analyze_module.job_manager, monkeypatch)
    user = _make_user(symbols_db)

    result = start_single_analysis(
        request=_fake_request(), mode="wachstumswerte", symbol="brk.b",
        frequency="annual", current_user=user, db=symbols_db,
    )

    assert result["symbol"] == "BRK-B"


# ---------------------------------------------------------------------------
# start_full_analysis (api/routes/full_analysis.py)
# ---------------------------------------------------------------------------

def test_start_full_analysis_accepts_known_active_symbol(symbols_db, monkeypatch):
    _isolate_job_manager(full_analysis_module.jobs, monkeypatch)
    user = _make_user(symbols_db)

    result = start_full_analysis(request=_fake_request(), symbol="AAPL", current_user=user, db=symbols_db)

    assert result["symbol"] == "AAPL"


def test_start_full_analysis_rejects_unknown_symbol_with_422(symbols_db, monkeypatch):
    monkeypatch.setattr(full_analysis_module.jobs, "submit", MagicMock())
    user = _make_user(symbols_db)

    with pytest.raises(HTTPException) as exc:
        start_full_analysis(request=_fake_request(), symbol="FOOBARX", current_user=user, db=symbols_db)
    assert exc.value.status_code == 422


def test_start_full_analysis_rejects_delisted_symbol(symbols_db, monkeypatch):
    monkeypatch.setattr(full_analysis_module.jobs, "submit", MagicMock())
    user = _make_user(symbols_db)

    with pytest.raises(HTTPException) as exc:
        start_full_analysis(request=_fake_request(), symbol="DEADCO", current_user=user, db=symbols_db)
    assert exc.value.status_code == 422
    assert "nicht mehr gelistet" in exc.value.detail


# ---------------------------------------------------------------------------
# start_custom_analysis (api/routes/custom_analysis.py)
# ---------------------------------------------------------------------------

def test_start_custom_analysis_accepts_known_active_symbol(symbols_db, monkeypatch):
    _isolate_job_manager(custom_analysis_module.job_manager, monkeypatch)
    user = _make_user(symbols_db)
    payload = CustomAnalysisStartRequest(symbol="AAPL", metrics=[{"key": "calculate_KGV", "params": {}}])

    result = start_custom_analysis(request=_fake_request(), payload=payload, current_user=user, db=symbols_db)

    assert result["symbol"] == "AAPL"


def test_start_custom_analysis_rejects_unknown_symbol_with_422(symbols_db, monkeypatch):
    monkeypatch.setattr(custom_analysis_module.job_manager, "submit", MagicMock())
    user = _make_user(symbols_db)
    payload = CustomAnalysisStartRequest(symbol="FOOBARX", metrics=[{"key": "calculate_KGV", "params": {}}])

    with pytest.raises(HTTPException) as exc:
        start_custom_analysis(request=_fake_request(), payload=payload, current_user=user, db=symbols_db)
    assert exc.value.status_code == 422


# ---------------------------------------------------------------------------
# run_definition (api/routes/custom_analysis.py)
# ---------------------------------------------------------------------------

def test_run_definition_accepts_known_active_symbol(symbols_db, monkeypatch):
    from api.crud import custom_analysis_definition as definitions_crud

    _isolate_job_manager(custom_analysis_module.job_manager, monkeypatch)
    user = _make_user(symbols_db)
    definition = definitions_crud.create_definition(
        symbols_db, user.id, "Meine Analyse", [{"key": "calculate_KGV", "params": {}}]
    )
    payload = CustomAnalysisDefinitionRunRequest(symbol="AAPL")

    result = run_definition(
        request=_fake_request(), definition_id=definition.id, payload=payload,
        current_user=user, db=symbols_db,
    )

    assert result["symbol"] == "AAPL"


def test_run_definition_rejects_unknown_symbol_with_422(symbols_db, monkeypatch):
    from api.crud import custom_analysis_definition as definitions_crud

    monkeypatch.setattr(custom_analysis_module.job_manager, "submit", MagicMock())
    user = _make_user(symbols_db)
    definition = definitions_crud.create_definition(
        symbols_db, user.id, "Meine Analyse", [{"key": "calculate_KGV", "params": {}}]
    )
    payload = CustomAnalysisDefinitionRunRequest(symbol="FOOBARX")

    with pytest.raises(HTTPException) as exc:
        run_definition(
            request=_fake_request(), definition_id=definition.id, payload=payload,
            current_user=user, db=symbols_db,
        )
    assert exc.value.status_code == 422
