"""EVOLVING.md EV-101: API-Ebenen-Ergänzung zum Golden Master in
agent/tests/test_golden_master_frequencies.py.

Ruft die synchronen Metrik-Endpoints aus api/routes/metric_routes.py direkt
als Python-Funktionen auf (Stil wie
api/tests/test_analysis_start_symbol_validation.py) statt über einen
FastAPI-TestClient - kein HTTP-Layer, keine Depends-Auflösung nötig, aber
exakt derselbe Funktionskörper, der auch die echte Route bedient.

Zusätzlich dokumentiert dieser Test einen wichtigen, in EVOLVING.md §4.3
noch nicht erfassten Befund: `POST /analyze/{mode}/start` validiert
`frequency` auf Routen-Ebene GAR NICHT - der Job läuft asynchron
(job_manager.submit), ein ungültiger Wert wird erst (falls überhaupt) tief
im Agenten sichtbar, nie als synchroner Fehler an den Aufrufer. EV-133 muss
diesen bestehenden Vertrag kennen, bevor es hier etwas validiert.
"""
from __future__ import annotations

from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from starlette.requests import Request

from agent.DataLoader import DataLoader
from agent.data_sources.sec_source import SecSource
from api.models.analysis_history import AnalysisHistory
from api.models.base import Base
from api.models.custom_analysis_definition import CustomAnalysisDefinition
from api.models.symbol import Symbol
from api.models.user import User
from api.routes import analyze as analyze_module
from api.routes import metric_routes

_FIXED_CURRENT_PRICE = 200.0


@pytest.fixture(autouse=True)
def deterministic_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dieselbe Determinismus-Strategie wie
    agent/tests/test_golden_master_frequencies.py (siehe dort für die
    Begründung): Datei-Cache nie als abgelaufen behandeln, aktuellen Kurs
    fest verdrahten, Netzwerk hart blockieren."""
    forever = timedelta(days=36500)
    monkeypatch.setattr(DataLoader, "_cache_duration_for", lambda self, data_type: forever)
    monkeypatch.setattr(DataLoader, "get_current_price_per_share", lambda self, symbol: _FIXED_CURRENT_PRICE)

    original_sec_load = SecSource._load_cached_data

    def _sec_load_ignoring_max_age(self, cache_key, max_age=forever):  # noqa: ANN001
        return original_sec_load(self, cache_key, max_age=forever)

    monkeypatch.setattr(SecSource, "_load_cached_data", _sec_load_ignoring_max_age)

    import requests
    import yfinance as yf

    def _block(*_a, **_kw):
        raise AssertionError("Golden-Master-API-Test hat einen echten Netzwerkzugriff ausgelöst.")

    monkeypatch.setattr(requests, "get", _block)
    monkeypatch.setattr(requests, "post", _block)
    monkeypatch.setattr(requests.Session, "get", _block)
    monkeypatch.setattr(requests.Session, "post", _block)
    monkeypatch.setattr(yf.Ticker, "history", _block, raising=False)


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


@pytest.fixture
def symbols_db(db):
    Base.metadata.create_all(
        db.get_bind(),
        tables=[Symbol.__table__, AnalysisHistory.__table__, CustomAnalysisDefinition.__table__],
    )
    db.add(Symbol(symbol="AAPL", name="Apple Inc.", exchange="NASDAQ", is_active=True))
    db.commit()
    return db


def _make_user(db) -> User:
    user = User(
        email="golden-master@example.com", hashed_password="x", plan="pro",
        email_verified=True, monthly_request_limit=None, monthly_request_count=0,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


# ---------------------------------------------------------------------------
# Synchrone Metrik-Endpoints: annual/quarterly/invalid - exakter Response-Text
# ---------------------------------------------------------------------------

def test_kuv_endpoint_annual_and_quarterly_and_invalid_frequency():
    request = _fake_request()

    annual = metric_routes.kuv(request, symbol="aapl", frequency="annual", current_user=None)
    quarterly = metric_routes.kuv(request, symbol="aapl", frequency="quarterly", current_user=None)
    invalid = metric_routes.kuv(request, symbol="aapl", frequency="invalid", current_user=None)

    assert annual["symbol"] == "AAPL"
    assert "error" not in annual
    assert "error" not in quarterly
    # Dokumentierter Ist-Zustand (siehe agent-seitigen Golden Master): kein
    # Guard auf Model-Ebene - "invalid" faellt in denselben Zweig wie
    # "quarterly" und liefert denselben KUV-Wert, keinen Fehler.
    assert invalid["KUV"] == quarterly["KUV"]


def test_net_debt_to_ebitda_endpoint_invalid_frequency_has_explicit_guard():
    """calculate_net_debt_to_ebitda reicht `frequency` an DataLoader-Methoden
    mit explizitem Guard durch - hier MUSS "invalid" als Fehler-Dict
    zurückkommen (Gegenbeispiel zu kuv oben)."""
    request = _fake_request()

    result = metric_routes.net_debt_to_ebitda(request, symbol="aapl", frequency="invalid", current_user=None)

    assert "error" in result
    assert "Ungültige Frequenz" in result["error"]


# ---------------------------------------------------------------------------
# start_single_analysis: dokumentiert, dass die Route `frequency` NICHT
# synchron validiert (Job läuft asynchron) - Grundlage für EV-133.
# ---------------------------------------------------------------------------

def test_start_single_analysis_accepts_any_frequency_string_synchronously(symbols_db, monkeypatch):
    monkeypatch.setattr(analyze_module.job_manager, "submit", MagicMock())
    monkeypatch.setattr(analyze_module.job_manager, "count_active_jobs", lambda user_id: 0)
    monkeypatch.setattr(analyze_module.job_manager, "create_job", MagicMock(return_value="test-job-id"))
    user = _make_user(symbols_db)

    result = analyze_module.start_single_analysis(
        request=_fake_request(), mode="wachstumswerte", symbol="AAPL",
        frequency="ttm-does-not-exist-yet", current_user=user, db=symbols_db,
    )

    # Kein 4xx/Exception - die Route nimmt jeden String an und reicht ihn nur
    # durch; die eigentliche Prüfung (falls vorhanden) passiert erst tief im
    # asynchronen Job (hier per Mock gar nicht ausgeführt).
    assert result["frequency"] == "ttm-does-not-exist-yet"
    assert result["job_id"] == "test-job-id"
