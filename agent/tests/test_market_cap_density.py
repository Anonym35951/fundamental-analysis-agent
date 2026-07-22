"""EVOLVING.md „Dichte historische Marktkapitalisierung" (MC-004):
`calculate_historical_market_cap` reindizierte den Kurs bisher auf die
quartalsweisen SEC-Berichtsstichtage - über "Max" (~20 Jahre) ~80 Punkte
(sah gut aus), aber 1y hatte nur ~2 Punkte (nur Start/Ende, kein Verlauf).
Jetzt: täglicher Kurs (Alpha-Vantage primär, Yahoo-Fallback) × forward-
gefüllte Aktienanzahl, danach altersgestuft ausgedünnt.

Kein Netzwerkzugriff: `requests.get` und `yfinance` werden gemockt (Muster
aus test_live_price_retry.py / test_historical_data_retry.py).
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from agent.DataLoader import DataLoader
from agent.Model import Model


@pytest.fixture
def loader():
    return DataLoader()


@pytest.fixture
def model():
    return Model()


# ---------------------------------------------------------------------------
# DataLoader.get_daily_price_series / _fetch_alpha_vantage_daily_series:
# Alpha-Vantage primär (Premium-Key), Yahoo als reiner Resilienz-Fallback.
# ---------------------------------------------------------------------------

def _av_daily_adjusted_response(dates_and_closes: dict[str, float]) -> MagicMock:
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "Time Series (Daily)": {
            date: {
                "1. open": str(close),
                "2. high": str(close),
                "3. low": str(close),
                "4. close": str(close),
                "5. adjusted close": str(close),
                "6. volume": "1000",
            }
            for date, close in dates_and_closes.items()
        }
    }
    return mock_response


def test_get_daily_price_series_uses_alpha_vantage_when_available(loader):
    mock_response = _av_daily_adjusted_response({"2024-01-02": 100.0, "2024-01-03": 101.5})

    with patch("agent.DataLoader.requests.get", return_value=mock_response) as mock_get, \
            patch("agent.DataLoader.yf.Ticker") as mock_yf:
        series = loader.get_daily_price_series("AAPL", use_cache=False)

    assert series is not None
    assert list(series.values) == [100.0, 101.5]
    assert series.index[0] == pd.Timestamp("2024-01-02")
    mock_get.assert_called_once()
    _, kwargs = mock_get.call_args
    assert kwargs["params"]["function"] == "TIME_SERIES_DAILY_ADJUSTED"
    assert kwargs["params"]["outputsize"] == "full"
    mock_yf.assert_not_called()


def test_get_daily_price_series_falls_back_to_yahoo_on_av_rate_limit(loader):
    """AV-Rate-Limit-Antworten kommen als HTTP 200 mit "Note"/"Information" -
    kein Crash, sondern ein sauberer Fallback auf die produktiv bewährte
    Yahoo-Tagesquelle (reine Resilienz, nicht wegen eines Entitlement-
    Zweifels - der Account hat einen AV-Premium-Key)."""
    mock_av_response = MagicMock()
    mock_av_response.json.return_value = {"Note": "Thank you for using Alpha Vantage! ..."}

    fake_yahoo_df = pd.DataFrame(
        {"Close": [55.0, 56.5]},
        index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
    )

    with patch("agent.DataLoader.requests.get", return_value=mock_av_response) as mock_get, \
            patch.object(loader, "get_max_historical_stock_data", return_value=fake_yahoo_df) as mock_yahoo:
        series = loader.get_daily_price_series("AAPL", use_cache=False)

    assert series is not None
    assert list(series.values) == [55.0, 56.5]
    mock_get.assert_called_once()
    mock_yahoo.assert_called_once_with("AAPL", use_cache=False, interval="1d")


def test_get_daily_price_series_returns_none_when_both_sources_fail(loader):
    mock_av_response = MagicMock()
    mock_av_response.json.return_value = {"Error Message": "Invalid API call"}

    with patch("agent.DataLoader.requests.get", return_value=mock_av_response), \
            patch.object(loader, "get_max_historical_stock_data", return_value=None):
        series = loader.get_daily_price_series("BROKEN", use_cache=False)

    assert series is None


def test_get_daily_price_series_falls_back_when_av_raises(loader):
    fake_yahoo_df = pd.DataFrame({"Close": [10.0]}, index=pd.to_datetime(["2024-01-02"]))

    with patch("agent.DataLoader.requests.get", side_effect=ConnectionError("AV unreachable")), \
            patch.object(loader, "get_max_historical_stock_data", return_value=fake_yahoo_df):
        series = loader.get_daily_price_series("AAPL", use_cache=False)

    assert series is not None
    assert list(series.values) == [10.0]


# ---------------------------------------------------------------------------
# Model._age_tiered_downsample: täglich <=2J / wöchentlich 2-5J / monatlich >5J.
# ---------------------------------------------------------------------------

def test_age_tiered_downsample_keeps_recent_daily_points_unchanged(model):
    anchor = pd.Timestamp("2024-01-01")
    recent_idx = pd.date_range(anchor - pd.Timedelta(days=10), anchor, freq="D")
    df = pd.DataFrame({"Value": range(len(recent_idx))}, index=recent_idx)

    result = model._age_tiered_downsample(df)

    assert len(result) == len(recent_idx)
    assert list(result.index) == list(recent_idx)
    assert list(result["Value"]) == list(range(len(recent_idx)))


def test_age_tiered_downsample_reduces_mid_range_to_weekly(model):
    # EVOLVING.md-Detail: der Anker ist `df.index.max()`, NICHT ein externes
    # "heute" - ein isoliertes Mid-Segment ohne einen tatsaechlich aktuellen
    # Punkt haette hier seinen eigenen (aelteren) letzten Tag als Anker
    # verwendet und waere faelschlich komplett als "recent" durchgereicht
    # worden. Ein zusaetzlicher Punkt GENAU am gewuenschten Anker-Datum
    # verankert den Test korrekt.
    anchor = pd.Timestamp("2024-01-01")
    # ~356 taegliche Punkte, komplett in der 2-5-Jahre-Zone (zw. 2 und 3 Jahre alt).
    mid_idx = pd.date_range(
        anchor - pd.DateOffset(years=3), anchor - pd.DateOffset(years=2, days=10), freq="D"
    )
    full_idx = mid_idx.append(pd.DatetimeIndex([anchor]))
    df = pd.DataFrame({"Value": range(len(full_idx))}, index=full_idx)

    result = model._age_tiered_downsample(df)

    assert len(mid_idx) > 300
    # Woechentlich (~52 Punkte fuer das Mid-Segment) + der eine Anker-Punkt
    # selbst (faellt in "recent", bleibt unveraendert) - deutlich weniger als
    # die 356 taeglichen Eingabepunkte, aber nicht auf einen Bruchteil eines
    # Punktes/Monat zusammengeschrumpft.
    assert 48 <= len(result) <= 56


def test_age_tiered_downsample_reduces_old_range_to_monthly(model):
    anchor = pd.Timestamp("2024-01-01")
    # 2 Jahre taegliche Punkte, komplett > 5 Jahre alt; + 1 Anker-Punkt (siehe
    # Kommentar oben) damit df.index.max() tatsaechlich "heute" ist.
    old_idx = pd.date_range(anchor - pd.DateOffset(years=8), anchor - pd.DateOffset(years=6), freq="D")
    full_idx = old_idx.append(pd.DatetimeIndex([anchor]))
    df = pd.DataFrame({"Value": range(len(full_idx))}, index=full_idx)

    result = model._age_tiered_downsample(df)

    assert len(old_idx) > 700
    assert 22 <= len(result) <= 28


def test_age_tiered_downsample_combines_all_three_segments_sorted_and_unique(model):
    anchor = pd.Timestamp("2024-06-15")
    recent_idx = pd.date_range(anchor - pd.Timedelta(days=5), anchor, freq="D")
    mid_idx = pd.date_range(anchor - pd.DateOffset(years=4), anchor - pd.DateOffset(years=3), freq="D")
    old_idx = pd.date_range(anchor - pd.DateOffset(years=10), anchor - pd.DateOffset(years=8), freq="D")
    full_idx = old_idx.append(mid_idx).append(recent_idx)
    df = pd.DataFrame({"Value": range(len(full_idx))}, index=full_idx)

    result = model._age_tiered_downsample(df)

    assert result.index.is_monotonic_increasing
    assert not result.index.duplicated().any()
    # Deutlich weniger Punkte als die ~1100 taeglichen Eingabepunkte (6 recent
    # + ~52 woechentlich + ~25 monatlich erwartet), aber mehr als nur ein
    # Punkt je Segment.
    assert len(full_idx) > 1000
    assert 70 <= len(result) <= 100


def test_age_tiered_downsample_empty_input_returns_empty(model):
    empty_df = pd.DataFrame({"Value": []}, index=pd.DatetimeIndex([]))
    result = model._age_tiered_downsample(empty_df)
    assert result.empty


# ---------------------------------------------------------------------------
# Model.calculate_historical_market_cap: Kurs x ffill(Aktienanzahl), Spalten-
# Contract (MarketCap zuerst) erhalten, Tage vor dem ersten Bilanzstichtag
# verworfen.
# ---------------------------------------------------------------------------

def _fake_balance_sheet(shares_by_date: dict[str, float]) -> pd.DataFrame:
    df = pd.DataFrame(
        {"commonStockSharesOutstanding": list(shares_by_date.values())},
        index=pd.to_datetime(list(shares_by_date.keys())),
    )
    return df


def test_calculate_historical_market_cap_multiplies_price_by_forward_filled_shares(model):
    balance_sheet = _fake_balance_sheet({"2024-01-02": 1_000_000, "2024-01-04": 1_100_000})
    daily_close = pd.Series(
        {"2024-01-02": 10.0, "2024-01-03": 11.0, "2024-01-04": 12.0, "2024-01-05": 13.0},
        dtype="float64",
    )
    daily_close.index = pd.to_datetime(daily_close.index)

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"balance_sheet": balance_sheet, "income_statement": None, "cash_flow": None},
    ), patch.object(model.dataloader, "get_daily_price_series", return_value=daily_close):
        df = model.calculate_historical_market_cap("TEST", use_cache=False)

    assert df is not None
    assert list(df.columns) == ["MarketCap", "commonStockSharesOutstanding", "Close"]
    # 2024-01-02/03: 1,000,000 Shares (ffill) -- 2024-01-04/05: 1,100,000 Shares.
    assert df.loc["2024-01-02", "MarketCap"] == 1_000_000 * 10.0
    assert df.loc["2024-01-03", "MarketCap"] == 1_000_000 * 11.0
    assert df.loc["2024-01-04", "MarketCap"] == 1_100_000 * 12.0
    assert df.loc["2024-01-05", "MarketCap"] == 1_100_000 * 13.0


def test_calculate_historical_market_cap_drops_days_before_first_report_date(model):
    balance_sheet = _fake_balance_sheet({"2024-01-03": 1_000_000})
    daily_close = pd.Series(
        {"2024-01-01": 9.0, "2024-01-02": 9.5, "2024-01-03": 10.0},
        dtype="float64",
    )
    daily_close.index = pd.to_datetime(daily_close.index)

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"balance_sheet": balance_sheet, "income_statement": None, "cash_flow": None},
    ), patch.object(model.dataloader, "get_daily_price_series", return_value=daily_close):
        df = model.calculate_historical_market_cap("TEST", use_cache=False)

    assert df is not None
    # Vor dem ersten Bilanzstichtag ist die Aktienanzahl unbekannt (ffill=NaN)
    # -> diese Tage duerfen nicht mit einer falschen (fehlenden) Anzahl in der
    # Serie landen.
    assert pd.Timestamp("2024-01-01") not in df.index
    assert pd.Timestamp("2024-01-02") not in df.index
    assert pd.Timestamp("2024-01-03") in df.index


def test_calculate_historical_market_cap_returns_none_when_no_daily_prices(model):
    balance_sheet = _fake_balance_sheet({"2024-01-02": 1_000_000})

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"balance_sheet": balance_sheet, "income_statement": None, "cash_flow": None},
    ), patch.object(model.dataloader, "get_daily_price_series", return_value=None):
        df = model.calculate_historical_market_cap("TEST", use_cache=False)

    assert df is None


# ---------------------------------------------------------------------------
# Model.calculate_historical_ev: EV muss weiterhin genau einen Wert je
# quartalsweisem Bilanzstichtag liefern - der alte `join(how="inner")` gegen
# einen jetzt taeglich/dicht indizierten market_cap_df wuerde fast immer leer
# laufen; die Fixversion nutzt einen "nächster vorheriger Wert"-Reindex
# (dieselbe Semantik wie zuvor intern in calculate_historical_market_cap).
# ---------------------------------------------------------------------------

def test_calculate_historical_ev_stays_quarterly_against_dense_market_cap(model):
    balance_sheet = pd.DataFrame(
        {
            "totalLiabilities": [500_000.0, 520_000.0],
            "cashAndCashEquivalentsAtCarryingValue": [100_000.0, 90_000.0],
        },
        index=pd.to_datetime(["2024-01-05", "2024-04-05"]),
    )

    # Dichte, NICHT auf die Bilanzstichtage ausgerichtete Marktkap-Serie -
    # genau das neue Verhalten von calculate_historical_market_cap.
    dense_dates = pd.date_range("2024-01-01", "2024-04-10", freq="D")
    market_cap_df = pd.DataFrame(
        {
            "MarketCap": [1_000_000.0 + i * 100 for i in range(len(dense_dates))],
            "commonStockSharesOutstanding": [10_000] * len(dense_dates),
            "Close": [100.0] * len(dense_dates),
        },
        index=dense_dates,
    )

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"balance_sheet": balance_sheet, "income_statement": None, "cash_flow": None},
    ), patch.object(model, "calculate_historical_market_cap", return_value=market_cap_df):
        df = model.calculate_historical_ev("TEST", use_cache=False)

    assert df is not None
    # Genau ein EV-Wert je Bilanzstichtag (quartalsweise Ausgabe erhalten).
    assert len(df) == 2
    assert list(df.index) == [pd.Timestamp("2024-01-05"), pd.Timestamp("2024-04-05")]

    expected_mc_jan = market_cap_df.loc["2024-01-05", "MarketCap"]
    expected_mc_apr = market_cap_df.loc["2024-04-05", "MarketCap"]
    assert df.loc["2024-01-05", "EV"] == expected_mc_jan + 500_000.0 - 100_000.0
    assert df.loc["2024-04-05", "EV"] == expected_mc_apr + 520_000.0 - 90_000.0


def test_calculate_historical_ev_uses_nearest_prior_market_cap_when_no_exact_date_match(model):
    """Bilanzstichtag faellt auf ein Wochenende / einen Nicht-Handelstag -
    die dichte Marktkap-Serie hat dafuer keinen exakten Eintrag; EV muss
    trotzdem den zeitlich naechsten VORHERGEHENDEN Wert uebernehmen statt
    leer zu laufen."""
    balance_sheet = pd.DataFrame(
        {
            "totalLiabilities": [500_000.0],
            "cashAndCashEquivalentsAtCarryingValue": [100_000.0],
        },
        index=pd.to_datetime(["2024-01-07"]),  # ein Sonntag
    )
    market_cap_df = pd.DataFrame(
        {
            "MarketCap": [1_000_000.0, 1_010_000.0],
            "commonStockSharesOutstanding": [10_000, 10_000],
            "Close": [100.0, 101.0],
        },
        index=pd.to_datetime(["2024-01-05", "2024-01-08"]),  # Freitag / Montag
    )

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"balance_sheet": balance_sheet, "income_statement": None, "cash_flow": None},
    ), patch.object(model, "calculate_historical_market_cap", return_value=market_cap_df):
        df = model.calculate_historical_ev("TEST", use_cache=False)

    assert df is not None
    assert len(df) == 1
    # ffill -> der letzte Wert VOR dem Sonntag ist Freitag (1,000,000), nicht
    # der spaetere Montagswert.
    assert df.loc["2024-01-07", "EV"] == 1_000_000.0 + 500_000.0 - 100_000.0
