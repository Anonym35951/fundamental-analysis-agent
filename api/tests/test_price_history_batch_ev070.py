"""Regressionstests für EVOLVING.md EV-070: GET /metrics/price-history-batch.

Stil wie test_price_history_ev060.py: direkter Funktionsaufruf gegen die
Route-Funktion (kein TestClient), _fake_request() für den
@limiter.limit-Dekorator, get_action() gemockt (kein echter yfinance-Call).
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.requests import Request

import api.routes.metric_routes as metric_routes
from api.core.rate_limit import limiter
from api.routes.metric_routes import price_history_batch


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/metrics/price-history-batch",
        "headers": [],
        "client": ("testclient", 12345),
        "server": ("testserver", 80),
        "scheme": "http",
        "query_string": b"",
    }
    return Request(scope)


def _daily_closes(start: str, periods: int, start_price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq="D")
    closes = [start_price + i for i in range(periods)]
    return pd.DataFrame({"Close": closes}, index=index)


@pytest.fixture(autouse=True)
def _isolate_price_history_cache(monkeypatch):
    monkeypatch.setattr(metric_routes, "_PRICE_HISTORY_CACHE", {})
    # Tests sollen nicht wirklich 200ms je Fetch schlafen.
    monkeypatch.setattr(metric_routes.time, "sleep", MagicMock())
    # price_history_batch hat ein Limit von 10/minute; alle Tests hier teilen
    # sich denselben Fake-Client ("testclient") und würden sich sonst
    # gegenseitig den Bucket leeren (429 statt des erwarteten Verhaltens).
    limiter.reset()


def _mock_action(per_symbol: dict):
    action = MagicMock()

    _MISSING = object()

    def _fake_fetch(symbol, use_cache=True, interval="1d"):
        result = per_symbol.get(symbol, _MISSING)
        if result is _MISSING or (isinstance(result, str) and result == "MISSING"):
            return None
        if isinstance(result, Exception):
            raise result
        return result

    action.model.dataloader.get_max_historical_stock_data = MagicMock(side_effect=_fake_fetch)
    return action


def _call(monkeypatch, per_symbol, symbols="AAPL,MSFT,KO", range="1m"):
    action = _mock_action(per_symbol)
    monkeypatch.setattr(metric_routes, "get_action", lambda: action)
    result = price_history_batch(request=_fake_request(), symbols=symbols, range=range, current_user=object())
    return result, action


def test_returns_one_result_per_symbol_in_order(monkeypatch):
    df = _daily_closes("2024-01-01", 20)
    result, _ = _call(monkeypatch, {"AAPL": df, "MSFT": df, "KO": df})

    assert [r["symbol"] for r in result["results"]] == ["AAPL", "MSFT", "KO"]
    for r in result["results"]:
        assert r["currency"] == "USD"
        assert len(r["rows"]) == 20


def test_normalizes_symbol_case_and_whitespace(monkeypatch):
    df = _daily_closes("2024-01-01", 10)
    result, action = _call(monkeypatch, {"AAPL": df}, symbols=" aapl ")
    assert result["results"][0]["symbol"] == "AAPL"
    action.model.dataloader.get_max_historical_stock_data.assert_called_once_with("AAPL", use_cache=True, interval="1d")


def test_partial_failure_yields_error_entry_without_failing_the_whole_batch(monkeypatch):
    df = _daily_closes("2024-01-01", 20)
    result, _ = _call(monkeypatch, {"AAPL": df, "MSFT": "MISSING", "KO": df})

    by_symbol = {r["symbol"]: r for r in result["results"]}
    assert "rows" in by_symbol["AAPL"]
    assert "rows" in by_symbol["KO"]
    assert "error" in by_symbol["MSFT"]
    assert "rows" not in by_symbol["MSFT"]


def test_dataloader_exception_yields_error_entry(monkeypatch):
    df = _daily_closes("2024-01-01", 20)
    result, _ = _call(monkeypatch, {"AAPL": df, "MSFT": RuntimeError("yfinance boom")})

    by_symbol = {r["symbol"]: r for r in result["results"]}
    assert "rows" in by_symbol["AAPL"]
    assert "error" in by_symbol["MSFT"]


def test_empty_symbols_raises_422(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _call(monkeypatch, {}, symbols="")
    assert exc_info.value.status_code == 422


def test_blank_symbols_after_stripping_raises_422(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _call(monkeypatch, {}, symbols=" , , ")
    assert exc_info.value.status_code == 422


def test_more_than_20_symbols_raises_422(monkeypatch):
    symbols = ",".join(f"SYM{i}" for i in range(21))
    with pytest.raises(HTTPException) as exc_info:
        _call(monkeypatch, {}, symbols=symbols)
    assert exc_info.value.status_code == 422
    assert "20" in exc_info.value.detail


def test_exactly_20_symbols_is_allowed(monkeypatch):
    df = _daily_closes("2024-01-01", 5)
    symbols = ",".join(f"SYM{i}" for i in range(20))
    per_symbol = {f"SYM{i}": df for i in range(20)}
    result, _ = _call(monkeypatch, per_symbol, symbols=symbols)
    assert len(result["results"]) == 20


def test_invalid_range_raises_422(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _call(monkeypatch, {"AAPL": _daily_closes("2024-01-01", 5)}, symbols="AAPL", range="1y")
    assert exc_info.value.status_code == 422


def test_second_call_hits_cache_for_already_fetched_symbol(monkeypatch):
    df = _daily_closes("2024-01-01", 20)
    action = _mock_action({"AAPL": df})
    monkeypatch.setattr(metric_routes, "get_action", lambda: action)

    price_history_batch(request=_fake_request(), symbols="AAPL", range="1m", current_user=object())
    price_history_batch(request=_fake_request(), symbols="AAPL", range="1m", current_user=object())

    action.model.dataloader.get_max_historical_stock_data.assert_called_once()


def test_throttles_only_between_uncached_fetches_not_before_the_first(monkeypatch):
    df = _daily_closes("2024-01-01", 5)
    result, _ = _call(monkeypatch, {"AAPL": df, "MSFT": df, "KO": df}, symbols="AAPL,MSFT,KO")

    assert len(result["results"]) == 3
    # 3 Symbole, alle ungecacht -> 2 Sleeps zwischen den Fetches (nicht vor
    # dem allerersten).
    assert metric_routes.time.sleep.call_count == 2


def test_no_throttle_when_all_but_one_symbol_are_cached(monkeypatch):
    df = _daily_closes("2024-01-01", 5)
    action = _mock_action({"AAPL": df, "MSFT": df, "KO": df})
    monkeypatch.setattr(metric_routes, "get_action", lambda: action)

    # Erster Aufruf fuellt den Cache fuer alle drei.
    price_history_batch(request=_fake_request(), symbols="AAPL,MSFT,KO", range="1m", current_user=object())
    metric_routes.time.sleep.reset_mock()

    # Zweiter Aufruf: alle drei bereits gecacht -> kein Sleep noetig.
    price_history_batch(request=_fake_request(), symbols="AAPL,MSFT,KO", range="1m", current_user=object())
    metric_routes.time.sleep.assert_not_called()
