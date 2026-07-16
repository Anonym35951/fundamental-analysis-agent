"""EVOLVING.md EV-133: API-Schicht für 'ttm'.

1) api/services/metric_catalog.get_catalog_entries() muss "ttm" im
   frequency-Param-enum anbieten und pro Metrik ein `supports_ttm`-Flag
   liefern, das exakt agent/frequency.py::TTM_CAPABLE_METRICS widerspiegelt.
2) /metrics/profit-growth (AAGR/AQGR) darf "ttm" NICHT stillschweigend als
   quarterly behandeln - siehe agent/frequency.py TTM_CAPABLE_METRICS-
   Docstring (Wachstumsraten sind explizit ausgeschlossen).
"""
from starlette.requests import Request

from agent.frequency import TTM_CAPABLE_METRICS
from api.routes import metric_routes
from api.services.metric_catalog import METRIC_CATALOG, get_catalog_entries


def _fake_request() -> Request:
    scope = {
        "type": "http", "method": "GET", "path": "/", "headers": [],
        "client": ("testclient", 12345), "server": ("testserver", 80),
        "scheme": "http", "query_string": b"",
    }
    return Request(scope)


def test_catalog_frequency_param_offers_ttm():
    entries = get_catalog_entries()
    entries_with_frequency = [e for e in entries if any(p["name"] == "frequency" for p in e["params"])]
    assert entries_with_frequency, "keine Metrik mit frequency-Param gefunden"
    for entry in entries_with_frequency:
        freq_param = next(p for p in entry["params"] if p["name"] == "frequency")
        assert freq_param["enum_values"] == ["annual", "quarterly", "ttm"]
        assert freq_param["default"] == "annual"


def test_catalog_supports_ttm_matches_capability_map_exactly():
    entries_by_key = {e["key"]: e for e in get_catalog_entries()}
    assert set(entries_by_key.keys()) == {spec.key for spec in METRIC_CATALOG}

    for key, entry in entries_by_key.items():
        assert entry["supports_ttm"] == (key in TTM_CAPABLE_METRICS), key


def test_profit_growth_rejects_ttm_instead_of_silently_using_quarterly(monkeypatch):
    called = {}

    def _fake_annual(symbol, start_date, end_date, use_cache):
        called["annual"] = True
        return {"avg_growth": 5.0}

    def _fake_quarterly(symbol, start_date, end_date, use_cache):
        called["quarterly"] = True
        return {"avg_growth": 5.0}

    action = metric_routes.get_action()
    monkeypatch.setattr(action.model, "calculate_avg_annual_profit_growth", _fake_annual)
    monkeypatch.setattr(action.model, "calculate_avg_quarterly_profit_growth", _fake_quarterly)

    result = metric_routes.profit_growth(_fake_request(), symbol="AAPL", frequency="ttm", current_user=None)

    assert "error" in result
    assert "Ungültige Frequenz" in result["error"]
    assert called == {}, "weder AAGR noch AQGR duerfen fuer 'ttm' aufgerufen werden"


def test_profit_growth_annual_and_quarterly_still_work_unchanged(monkeypatch):
    action = metric_routes.get_action()
    monkeypatch.setattr(action.model, "calculate_avg_annual_profit_growth", lambda *a, **kw: {"avg_growth": 7.0})
    monkeypatch.setattr(action.model, "calculate_avg_quarterly_profit_growth", lambda *a, **kw: {"avg_growth": 3.0})

    annual_result = metric_routes.profit_growth(_fake_request(), symbol="AAPL", frequency="annual", current_user=None)
    quarterly_result = metric_routes.profit_growth(_fake_request(), symbol="AAPL", frequency="quarterly", current_user=None)

    assert annual_result == {"avg_growth": 7.0}
    assert quarterly_result == {"avg_growth": 3.0}
