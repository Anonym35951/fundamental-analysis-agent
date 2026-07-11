"""Regressionstests für die TTM-Umstellung der historischen Preis-Multiples
(LAUNCH.md Block 5, 2026-07-11): calculate_historical_sales/-ebit/-ebitda
liefern jetzt eine rollierende 4-Quartals-Summe statt eines rohen
Einzelquartalswerts (der ~4x niedriger als die TTM-basierte Marktkonvention
war). Deckt die Rolling-Randfälle ab (Datenlücke, zu wenig Quartale,
Sortierrichtung) sowie die Konsistenz-Garantie an den drei Stellen, die eine
"aktuelle" Basisgröße für Kursziele/Bandbreiten-Bewertung brauchen - diese
müssen dieselbe TTM-Quelle nutzen wie die historische Reihe, sonst entsteht
ein neuer K-1/K-2-artiger Inkonsistenz-Bug (Zähler TTM, Nenner Einzelquartal).

Kein Netzwerkzugriff: get_fundamental_data wird gemockt.
"""
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _income_statement(quarters: dict, ascending: bool = False) -> pd.DataFrame:
    """quarters: {"YYYY-MM-DD": operatingIncome}."""
    dates = pd.to_datetime(list(quarters.keys()))
    df = pd.DataFrame({"operatingIncome": list(quarters.values())}, index=dates)
    return df.sort_index(ascending=ascending)


# ---------------------------------------------------------------------------
# Rolling-Randfälle (calculate_historical_ebit als Vertreter - dieselbe
# Rolling-Logik gilt identisch für calculate_historical_sales/-ebitda)
# ---------------------------------------------------------------------------

def test_ttm_ignores_sort_order_of_input(model):
    """Regressionsguard für den 'rollt rückwärts in der Zeit'-Fehler: ein
    aufsteigend UND ein absteigend sortiertes Eingabe-DataFrame mit
    identischen Werten müssen dasselbe TTM-Ergebnis liefern."""
    quarters = {
        "2023-03-31": 10.0, "2023-06-30": 20.0, "2023-09-30": 30.0, "2023-12-31": 40.0,
        "2024-03-31": 50.0,
    }

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"income_statement": _income_statement(quarters, ascending=False)},
    ):
        df_desc = model.calculate_historical_ebit("TEST", use_cache=False)

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"income_statement": _income_statement(quarters, ascending=True)},
    ):
        df_asc = model.calculate_historical_ebit("TEST", use_cache=False)

    assert df_desc.loc["2024-03-31", "EBIT"] == pytest.approx(140.0)  # 50+40+30+20
    assert df_asc.loc["2024-03-31", "EBIT"] == pytest.approx(140.0)


def test_ttm_requires_four_full_quarters_per_row(model):
    """Nur Zeilen mit vollständiger 4-Quartals-Trailing-Basis dürfen einen
    TTM-Wert haben - die ersten 3 Quartale einer Serie haben keine
    vollständige Trailing-Basis und müssen fehlen, nicht mit NaN
    durchgereicht werden."""
    quarters = {
        "2023-03-31": 10.0, "2023-06-30": 20.0, "2023-09-30": 30.0, "2023-12-31": 40.0,
        "2024-03-31": 50.0,
    }

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"income_statement": _income_statement(quarters)},
    ):
        df = model.calculate_historical_ebit("TEST", use_cache=False)

    assert set(df.index.strftime("%Y-%m-%d")) == {"2023-12-31", "2024-03-31"}
    assert not df["EBIT"].isna().any()


def test_ttm_returns_none_with_fewer_than_four_quarters(model):
    quarters = {"2023-03-31": 10.0, "2023-06-30": 20.0, "2023-09-30": 30.0}

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"income_statement": _income_statement(quarters)},
    ):
        df = model.calculate_historical_ebit("TEST", use_cache=False)

    assert df is None


def test_ttm_handles_gap_in_quarterly_data(model):
    """Eine fehlende Quartals-Periode mittendrin darf nicht stillschweigend
    übersprungen werden, als wäre sie da - rolling(4) muss die 4 tatsächlich
    VORHANDENEN Zeilen summieren, nicht 4 Kalenderquartale erzwingen (das ist
    bereits das bestehende Pandas-rolling-Verhalten, hier als expliziter
    Dokumentations-/Regressionstest)."""
    quarters = {
        "2023-03-31": 10.0, "2023-06-30": 20.0,
        # Lücke: Q3 2023 fehlt
        "2023-12-31": 40.0, "2024-03-31": 50.0,
    }

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"income_statement": _income_statement(quarters)},
    ):
        df = model.calculate_historical_ebit("TEST", use_cache=False)

    # Die 4 tatsächlich vorhandenen Zeilen (10+20+40+50), nicht 4
    # Kalenderquartale
    assert df.loc["2024-03-31", "EBIT"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Konsistenz-Guards: "aktuelle" Basisgröße muss aus derselben TTM-Quelle
# kommen wie die historische Reihe
# ---------------------------------------------------------------------------

def test_course_target_price_multiples_uses_ttm_sales_not_quarterly_revenue(model):
    historical_data = pd.DataFrame(
        {"Price_Sales": [1.0, 2.0, 3.0, 1.5, 2.5, 3.5]},
        index=pd.date_range("2019-01-01", periods=6, freq="YE"),
    )
    ttm_sales_df = pd.DataFrame({"Sales": [1_000_000.0]}, index=[pd.Timestamp("2024-12-31")])

    with patch.object(model.dataloader, "get_revenue") as mock_quarterly_revenue, \
         patch.object(model, "calculate_historical_sales", return_value=ttm_sales_df), \
         patch.object(model, "_resolve_shares_outstanding", return_value=100_000.0):
        result = model.calculate_course_target_PriceMultiples(historical_data, "TEST")

    assert "error" not in result
    mock_quarterly_revenue.assert_not_called()


def test_course_target_ev_multiples_uses_ttm_ebitda_not_quarterly(model):
    historical_data = pd.DataFrame(
        {"EV_EBITDA": [5.0, 8.0, 10.0, 6.0, 9.0, 11.0]},
        index=pd.date_range("2019-01-01", periods=6, freq="YE"),
    )
    ttm_ebitda_df = pd.DataFrame({"EBITDA": [2_000_000.0]}, index=[pd.Timestamp("2024-12-31")])

    with patch.object(model.dataloader, "get_ebitda_data") as mock_quarterly_ebitda, \
         patch.object(model, "calculate_historical_ebitda", return_value=ttm_ebitda_df), \
         patch.object(model, "calculate_buy_case", return_value={"buy_value": 6.0}), \
         patch.object(model, "calculate_worst_case", return_value=5.0), \
         patch.object(model, "calculate_sell_case", return_value=10.0), \
         patch.object(model, "calculate_fairValue_case", return_value=7.5), \
         patch.object(model.dataloader, "get_net_debt_data", return_value={"net_debt": 100_000.0}), \
         patch.object(model, "_resolve_shares_outstanding", return_value=100_000.0), \
         patch.object(model.dataloader, "get_minority_interest", return_value={"minority_interest": 0.0}), \
         patch.object(model.dataloader, "get_preferred_stock", return_value={"preferred_stock": 0.0}):
        result = model.calculate_course_target_EVMultiples("TEST", historical_data)

    assert "error" not in result
    mock_quarterly_ebitda.assert_not_called()


def test_evaluate_ebit_bandwidth_uses_ttm_ebit_not_quarterly(model):
    price_to_ebit_df = pd.DataFrame(
        {"Price_EBIT": [8.0] * 12, "Price": [100.0] * 12, "EBIT": [12.5] * 12},
        index=pd.date_range("2010-01-01", periods=12, freq="YE"),
    )
    ttm_ebit_df = pd.DataFrame({"EBIT": [500_000.0]}, index=[pd.Timestamp("2024-12-31")])
    price_df = pd.DataFrame({"Close": [100.0]}, index=[pd.Timestamp("2024-12-31")])

    with patch.object(model, "calculate_historical_price_to_ebit", return_value=price_to_ebit_df), \
         patch.object(model.dataloader, "get_ebit_data") as mock_quarterly_ebit, \
         patch.object(model, "calculate_historical_ebit", return_value=ttm_ebit_df), \
         patch.object(model, "_resolve_shares_outstanding", return_value=100_000.0), \
         patch.object(model.dataloader, "get_max_historical_stock_data", return_value=price_df):
        result = model.evaluate_ebit_bandwidth("TEST", min_years=5.0, use_cache=False)

    assert "error" not in result
    mock_quarterly_ebit.assert_not_called()
    assert "ebit_ratio" in result["current"]
