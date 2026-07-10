"""Regressionstest für den Live-Bug (2026-07-10): eine individuelle Analyse
mit der Ausschüttungsquote schlug online mit "HTTP 401" fehl, obwohl derselbe
Bug (P2-12: totes @retry, weil die Methode jede Exception intern fängt) für
get_current_price_per_share/get_max_historical_stock_data schon behoben war -
get_dividend_history und get_payout_ratio_data_annual hatten ihn noch.

Zusätzlich deckt dieser Test zwei weitere, unabhängige Bugs ab, beide live
gegen echte PYPL-Daten gefunden (nicht durch die anfänglichen, zu
optimistischen Mocks):
1. Der Fallback-Pfad (Schritt 3, SEC-basiert) nutzte das Ergebnis von
   get_shares_outstanding() direkt als Zahl, obwohl die Methode bei Erfolg
   ein Dict {"shares_outstanding": X, ...} zurückgibt - TypeError.
2. _sum_dividends_last_365_days: .sum() auf dem ganzen (einspaltigen)
   DataFrame statt der Spalte liefert eine 1-Element-Series statt eines
   Skalars; zusätzlich verliert der DatetimeIndex nach einem JSON-Cache-
   Roundtrip seine Zeitzone (dieselbe bekannte Ursache wie in
   Model.calculate_historical_dividend_yield_average) - ein Vergleich gegen
   ein tz-aware Timestamp crasht dann mit "Invalid comparison between
   dtype=datetime64[ns] and Timestamp".

Kein Netzwerkzugriff: yf.Ticker wird gemockt.
"""
import pandas as pd
import pytest
from tenacity import RetryError
from unittest.mock import MagicMock, patch

from agent.DataLoader import DataLoader


@pytest.fixture
def loader():
    return DataLoader()


# ---------------------------------------------------------------------------
# get_dividend_history
# ---------------------------------------------------------------------------

def _fake_dividends():
    return pd.Series([0.1, 0.1, 0.1, 0.1], index=pd.to_datetime(
        ["2025-01-01", "2025-04-01", "2025-07-01", "2025-10-01"]
    ))


def test_get_dividend_history_succeeds_without_retry(loader):
    fake_ticker = MagicMock()
    fake_ticker.dividends = _fake_dividends()

    with patch("agent.DataLoader.yf.Ticker", return_value=fake_ticker) as mock_ticker_cls:
        result = loader.get_dividend_history("AAPL", use_cache=False)

    assert "error" not in result
    assert mock_ticker_cls.call_count == 1


def test_get_dividend_history_retries_on_transient_failure(loader):
    fake_ticker = MagicMock()
    fake_ticker.dividends = _fake_dividends()

    call_count = {"n": 0}

    def flaky_ticker(symbol):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise ConnectionError("401 Unauthorized")
        return fake_ticker

    with patch("agent.DataLoader.yf.Ticker", side_effect=flaky_ticker):
        result = loader.get_dividend_history("MSFT", use_cache=False)

    assert "error" not in result
    assert call_count["n"] == 3


def test_get_dividend_history_raises_after_exhausting_retries(loader):
    with patch("agent.DataLoader.yf.Ticker", side_effect=ConnectionError("401 Unauthorized")) as mock_ticker_cls:
        with pytest.raises(RetryError) as exc_info:
            loader.get_dividend_history("TSLA", use_cache=False)

    assert isinstance(exc_info.value.__cause__, ConnectionError)
    assert mock_ticker_cls.call_count == 3


def test_get_dividend_history_empty_result_is_not_retried(loader):
    fake_ticker = MagicMock()
    fake_ticker.dividends = pd.Series(dtype=float)

    with patch("agent.DataLoader.yf.Ticker", return_value=fake_ticker) as mock_ticker_cls:
        result = loader.get_dividend_history("NODATA", use_cache=False)

    assert "error" in result
    assert mock_ticker_cls.call_count == 1


# ---------------------------------------------------------------------------
# get_payout_ratio_data_annual
# ---------------------------------------------------------------------------

def _fake_shares_result():
    return {"shares_outstanding": 1_000_000.0, "symbol": "PYPL", "date": "2025-12-31"}


def test_get_payout_ratio_uses_yahoo_payout_ratio_when_available(loader):
    with patch.object(loader, "get_shares_outstanding", return_value=_fake_shares_result()), \
            patch.object(loader, "_fetch_yahoo_info", return_value={"payoutRatio": 0.05}):
        result = loader.get_payout_ratio_data_annual("PYPL")

    assert "error" not in result
    assert result["payout_ratio_eps"] == 5.0
    assert result["shares_outstanding"] == 1_000_000.0  # entpackte Zahl, kein Dict


def test_get_payout_ratio_trailing_eps_path_returns_scalar_dps(loader):
    """Schritt 2 (trailingEps + Dividendenhistorie) war schon vor dem
    heutigen Fix regulär erreichbar (kein Yahoo-Ausfall nötig, nur fehlendes
    payoutRatio bei vorhandenem trailingEps) - dieselbe .sum()-auf-DataFrame-
    Falle wie in Schritt 3, hier unabhängig davon abgedeckt."""
    dividend_history = {
        "dividends_history": pd.DataFrame(
            {"dividend": [0.1, 0.1, 0.1, 0.1]},
            index=pd.date_range(end=pd.Timestamp.now(tz="America/New_York"), periods=4, freq="90D"),
        )
    }

    with patch.object(loader, "get_shares_outstanding", return_value=_fake_shares_result()), \
            patch.object(loader, "_fetch_yahoo_info", return_value={"trailingEps": 2.0}), \
            patch.object(loader, "get_dividend_history", return_value=dividend_history):
        result = loader.get_payout_ratio_data_annual("PYPL")

    assert "error" not in result
    assert isinstance(result["dps"], float)
    assert isinstance(result["payout_ratio_eps"], float)
    assert result["dps"] == 0.4
    assert result["payout_ratio_eps"] == 20.0  # 0.4 / 2.0 * 100


def test_get_payout_ratio_falls_back_to_sec_when_yahoo_info_exhausted(loader):
    """Live-Bug: Yahoo .info schlägt nach 3 Versuchen dauerhaft fehl (401) -
    darf NICHT die ganze Berechnung abbrechen, sondern muss auf die SEC-
    basierte Schritt-3-Berechnung zurückfallen, die kein Yahoo .info braucht."""
    financials = pd.DataFrame({pd.Timestamp("2025-12-31"): [100_000.0]}, index=["Net Income"])
    dividend_history = {
        "dividends_history": pd.DataFrame(
            {"dividend": [0.1, 0.1, 0.1, 0.1]},
            index=pd.date_range(end=pd.Timestamp.now(tz="America/New_York"), periods=4, freq="90D"),
        )
    }

    with patch.object(loader, "get_shares_outstanding", return_value=_fake_shares_result()), \
            patch.object(loader, "_fetch_yahoo_info", side_effect=RetryError(last_attempt=None)), \
            patch.object(loader, "get_stock_financials", return_value=financials), \
            patch.object(loader, "get_dividend_history", return_value=dividend_history):
        result = loader.get_payout_ratio_data_annual("PYPL")

    assert "error" not in result
    assert result["shares_outstanding"] == 1_000_000.0  # entpackte Zahl, kein Dict -> kein TypeError
    assert result["net_income"] == 100_000.0
    assert result["eps"] == 0.1  # 100_000 / 1_000_000
    # dividends_history hat nur eine Spalte ("dividend") - .sum() auf dem
    # ganzen (gefilterten) DataFrame statt der Spalte liefert eine
    # 1-Element-Series statt eines Skalars, was analyze_payout_ratio's
    # isinstance(value, (int, float))-Check zum Scheitern brachte (Live-Bug,
    # 2026-07-10: "Ungültige Ausschüttungsquote für PYPL: dividend    5.13
    # dtype: float64").
    assert isinstance(result["dps"], float)
    assert isinstance(result["payout_ratio_eps"], float)
    assert result["dps"] == 0.4  # 4 x 0.1
    assert result["payout_ratio_eps"] == 400.0  # 0.4 / 0.1 * 100


def test_get_payout_ratio_surfaces_invalid_shares_outstanding(loader):
    with patch.object(loader, "get_shares_outstanding", return_value={"shares_outstanding": None, "symbol": "X"}):
        result = loader.get_payout_ratio_data_annual("X")

    assert "error" in result


def test_sum_dividends_last_365_days_handles_tz_naive_cached_index(loader):
    """Reproduziert exakt den dritten Live-Bug: dividends_history nach einem
    Cache-Roundtrip ist tz-naiv, während pd.Timestamp.now(tz="America/New_York")
    tz-aware ist - ohne die tz_localize(None)-Normalisierung crasht der
    Vergleich statt einen Skalar zu liefern. Termine bewusst klar innerhalb
    des 365-Tage-Fensters (nicht direkt an der Grenze "jetzt"), damit der
    Test nicht von der wenige-Stunden-Verschiebung zwischen System-Ortszeit
    und America/New_York abhängt."""
    now = pd.Timestamp.now()
    tz_naive_history = pd.DataFrame(
        {"dividend": [0.1, 0.1, 0.1, 0.1]},
        index=[now - pd.Timedelta(days=d) for d in (300, 210, 120, 30)],  # keine tz
    )
    assert tz_naive_history.index.tz is None

    result = loader._sum_dividends_last_365_days(tz_naive_history)

    assert result == 0.4
    assert isinstance(result, float)
