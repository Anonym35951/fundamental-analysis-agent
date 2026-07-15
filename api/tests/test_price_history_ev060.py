"""Regressionstests für EVOLVING.md EV-060: GET /metrics/price-history/{symbol}.

Stil wie test_admin_symbols.py/test_symbol_search.py: direkter Funktionsaufruf
gegen die Route-Funktion (kein TestClient), _fake_request() für den
@limiter.limit-Dekorator, get_action() gemockt (kein echter yfinance-Call).
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi import HTTPException
from starlette.requests import Request

import api.routes.metric_routes as metric_routes
from api.routes.metric_routes import price_history


def _fake_request() -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/metrics/price-history/AAPL",
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
    # Der In-Process-Cache ist ein Modul-Singleton (wie _PRICE_CACHE) - ohne
    # Isolation würde Testlauf N den Cache-Eintrag von Testlauf N-1 treffen.
    monkeypatch.setattr(metric_routes, "_PRICE_HISTORY_CACHE", {})


def _mock_action(df):
    action = MagicMock()
    action.model.dataloader.get_max_historical_stock_data = MagicMock(return_value=df)
    return action


def _call(monkeypatch, df=None, action=None, symbol="AAPL", range="1y", raises=None):
    if action is None:
        action = _mock_action(df)
    monkeypatch.setattr(metric_routes, "get_action", lambda: action)
    if raises is not None:
        action.model.dataloader.get_max_historical_stock_data.side_effect = raises
    return price_history(request=_fake_request(), symbol=symbol, range=range, current_user=object()), action


def test_returns_expected_shape_and_currency(monkeypatch):
    df = _daily_closes("2024-01-01", 30)
    result, _ = _call(monkeypatch, df=df, range="max")

    assert result["symbol"] == "AAPL"
    assert result["currency"] == "USD"
    assert result["range"] == "max"
    assert len(result["rows"]) == 30
    assert result["rows"][0]["date"] == "2024-01-01"
    assert result["rows"][0]["close"] == 100.0
    assert result["rows"][-1]["date"] == "2024-01-30"


def test_uppercases_and_strips_symbol(monkeypatch):
    df = _daily_closes("2024-01-01", 5)
    result, _ = _call(monkeypatch, df=df, symbol=" aapl ", range="max")
    assert result["symbol"] == "AAPL"


def test_invalid_range_raises_422(monkeypatch):
    df = _daily_closes("2024-01-01", 5)
    with pytest.raises(HTTPException) as exc_info:
        _call(monkeypatch, df=df, range="3y")
    assert exc_info.value.status_code == 422


def test_missing_data_raises_404(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _call(monkeypatch, df=None, range="1y")
    assert exc_info.value.status_code == 404


def test_empty_dataframe_raises_404(monkeypatch):
    empty = pd.DataFrame({"Close": []}, index=pd.DatetimeIndex([]))
    with pytest.raises(HTTPException) as exc_info:
        _call(monkeypatch, df=empty, range="1y")
    assert exc_info.value.status_code == 404


def test_dataloader_exception_raises_502(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        _call(monkeypatch, df=None, range="1y", raises=RuntimeError("yfinance boom"))
    assert exc_info.value.status_code == 502


def test_range_cuts_relative_to_the_latest_available_date_not_today(monkeypatch):
    # Zwei Jahre taegliche Daten, Anker = letzter Punkt der Serie (nicht
    # "heute" - EVOLVING.md EV-060 spiegelt hier bewusst EV-040s
    # filterSeriesByRange-Vertrag).
    df = _daily_closes("2022-01-01", 730)
    result, _ = _call(monkeypatch, df=df, range="1m")

    dates = [row["date"] for row in result["rows"]]
    anchor = df.index.max().strftime("%Y-%m-%d")
    assert dates[-1] == anchor
    # 1 Monat taeglich sind grob 28-31 Punkte, sicher deutlich weniger als
    # die vollen 730 Tage.
    assert 25 <= len(dates) <= 32


def test_2y_range_is_downsampled_to_weekly(monkeypatch):
    df = _daily_closes("2020-01-01", 365 * 3)
    result, _ = _call(monkeypatch, df=df, range="2y")

    # ~2 Jahre woechentlich sind grob 104 Punkte, nicht die ~730 taeglichen
    # Punkte im 2y-Fenster.
    assert 90 <= len(result["rows"]) <= 115


def test_5y_range_is_downsampled_to_weekly(monkeypatch):
    df = _daily_closes("2010-01-01", 365 * 8)
    result, _ = _call(monkeypatch, df=df, range="5y")
    assert 250 <= len(result["rows"]) <= 275


def test_max_range_stays_daily_when_series_is_short(monkeypatch):
    df = _daily_closes("2024-01-01", 200)
    result, _ = _call(monkeypatch, df=df, range="max")
    assert len(result["rows"]) == 200


def test_max_range_is_downsampled_to_monthly_when_series_is_long(monkeypatch):
    df = _daily_closes("2000-01-01", 365 * 10)  # ~3650 taegliche Punkte
    result, _ = _call(monkeypatch, df=df, range="max")

    # ~10 Jahre monatlich sind grob 120 Punkte, weit unter den ~3650
    # taeglichen Rohpunkten und unter der 1500er-Schwelle.
    assert len(result["rows"]) < 150


def test_second_call_within_ttl_hits_cache_not_the_dataloader(monkeypatch):
    df = _daily_closes("2024-01-01", 30)
    action = _mock_action(df)
    monkeypatch.setattr(metric_routes, "get_action", lambda: action)

    first = price_history(request=_fake_request(), symbol="AAPL", range="max", current_user=object())
    second = price_history(request=_fake_request(), symbol="AAPL", range="max", current_user=object())

    assert first == second
    action.model.dataloader.get_max_historical_stock_data.assert_called_once()


def test_different_ranges_are_cached_independently(monkeypatch):
    df = _daily_closes("2024-01-01", 400)
    action = _mock_action(df)
    monkeypatch.setattr(metric_routes, "get_action", lambda: action)

    price_history(request=_fake_request(), symbol="AAPL", range="1m", current_user=object())
    price_history(request=_fake_request(), symbol="AAPL", range="max", current_user=object())

    assert action.model.dataloader.get_max_historical_stock_data.call_count == 2
