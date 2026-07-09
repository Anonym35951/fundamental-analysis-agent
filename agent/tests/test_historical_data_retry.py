"""Regressionstest: dieselbe tote-@retry-Ursache (LAUNCH_AUDIT.md P2-12) wie
bei `get_current_price_per_share`, hier für `get_max_historical_stock_data` -
die Datenquelle hinter praktisch jedem Chart/CRV/Bandbreiten-Feature
(TBV-/EBIT-Bandbreite, historische Multiples, Compare-Charts).

Kein Netzwerkzugriff: `yf.Ticker` wird gemockt.
"""
import pandas as pd
import pytest
from tenacity import RetryError
from unittest.mock import MagicMock, patch

from agent.DataLoader import DataLoader
from agent.Model import Model


def _fake_history_df():
    return pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [42.0], "Volume": [100]},
        index=pd.to_datetime(["2024-01-01"], utc=True),
    )


@pytest.fixture
def loader():
    return DataLoader()


def test_get_max_historical_stock_data_succeeds_without_retry(loader):
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = _fake_history_df()

    with patch("agent.DataLoader.yf.Ticker", return_value=fake_ticker) as mock_ticker_cls:
        df = loader.get_max_historical_stock_data("AAPL", use_cache=False)

    assert df is not None
    assert not df.empty
    assert mock_ticker_cls.call_count == 1


def test_get_max_historical_stock_data_retries_on_transient_failure(loader):
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = _fake_history_df()

    call_count = {"n": 0}

    def flaky_ticker(symbol):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("429 Too Many Requests")
        return fake_ticker

    with patch("agent.DataLoader.yf.Ticker", side_effect=flaky_ticker):
        df = loader.get_max_historical_stock_data("MSFT", use_cache=False)

    assert df is not None
    assert call_count["n"] == 3


def test_get_max_historical_stock_data_raises_after_exhausting_retries(loader):
    with patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("429 Too Many Requests")) as mock_ticker_cls:
        with pytest.raises(RetryError) as exc_info:
            loader.get_max_historical_stock_data("TSLA", use_cache=False)

    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert mock_ticker_cls.call_count == 3


def test_get_max_historical_stock_data_empty_result_is_not_retried(loader):
    """df.empty ist eine gültige "keine Daten"-Antwort (z. B. Symbol ohne
    Historie), kein transienter Fehler - darf NICHT wie eine Exception
    behandelt/wiederholt werden."""
    fake_ticker = MagicMock()
    fake_ticker.history.return_value = pd.DataFrame()

    with patch("agent.DataLoader.yf.Ticker", return_value=fake_ticker) as mock_ticker_cls:
        df = loader.get_max_historical_stock_data("NODATA", use_cache=False)

    assert df is None
    assert mock_ticker_cls.call_count == 1


# ---------------------------------------------------------------------------
# evaluate_tbv_bandwidth / evaluate_ebit_bandwidth: die beiden direkten
# Aufrufstellen ohne eigenes umschließendes try/except - müssen die jetzt
# propagierende Exception selbst abfangen statt ungeschützt crashen zu lassen.
# ---------------------------------------------------------------------------

@pytest.fixture
def model():
    return Model()


def test_evaluate_tbv_bandwidth_handles_historical_price_failure_gracefully(model):
    """evaluate_tbv_bandwidth hat kein eigenes umschließendes try/except -
    der direkte get_max_historical_stock_data-Aufruf (für den aktuellen
    Kurs) muss deshalb explizit abgesichert sein (siehe Fix in Model.py),
    sonst würde eine jetzt propagierende RetryError den kompletten Aufruf
    crashen lassen statt ein Error-Dict zurückzugeben."""
    pb_history = pd.DataFrame(
        {
            "Price_TangibleBookValue": [1.0] * 20,
            "Price": [10.0] * 20,
            "TangibleBookValue": [10.0] * 20,
            "commonStockSharesOutstanding": [1_000_000] * 20,
        },
        index=pd.date_range("2010-01-01", periods=20, freq="YE"),
    )

    with patch.object(
        model, "calculate_historical_price_to_TangibleBookValue", return_value=pb_history,
    ), patch.object(
        model.dataloader, "get_balance_sheet", return_value=pd.DataFrame({"2024-12-31": {}}),
    ), patch.object(
        model, "get_tangible_book_value", return_value=5_000_000.0,
    ), patch.object(
        model, "_resolve_shares_outstanding", return_value=1_000_000,
    ), patch.object(
        model.dataloader, "get_max_historical_stock_data",
        side_effect=RetryError(last_attempt=None),
    ):
        result = model.evaluate_tbv_bandwidth("AAPL", min_years=5.0)

    assert isinstance(result, dict)
    assert "error" in result
    assert "Kursdaten" in result["error"]
