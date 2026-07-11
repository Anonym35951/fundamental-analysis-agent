"""Regressionstests für die Kernberechnungen der Analyse-Engine (EBIT,
EBITDA, Free Cashflow, Tangible Book Value).

EBIT/EBITDA sind seit der TTM-Umstellung (2026-07-11, LAUNCH.md Block 5)
rollierende 4-Quartals-Summen, keine Einzelquartalswerte mehr (siehe
calculate_historical_ebit/-ebitda in Model.py) - die zugehörigen Cache-Keys
wurden dabei gebumpt (historical_ebit_ttm_/historical_ebitda_ttm_), daher
laufen die EBIT/EBITDA-Tests hier über synthetische, gemockte
get_fundamental_data-Eingaben statt über die eingefrorenen Cache-Fixtures
(die noch die alten, ungebumpten Cache-Keys tragen und ohnehin nur
Einzelquartalswerte enthalten). TBV/FreeCashflow/Bilanz-Tests sind von der
TTM-Umstellung nicht betroffen und lesen weiterhin aus den eingefrorenen
Fixtures (agent/tests/fixtures/cache/, siehe conftest.py::frozen_cache_dir).

Kein Netzwerkzugriff in beiden Fällen.
"""
import math

import pandas as pd
import pytest
from unittest.mock import patch


def _synthetic_income_statement(quarters: dict) -> pd.DataFrame:
    """quarters: {"YYYY-MM-DD": (operatingIncome, depreciationAndAmortization)},
    absteigend sortiert zurückgegeben (Alpha-Vantage-Konvention: neueste
    zuerst) - die TTM-Rolling-Logik in Model.py muss das intern selbst
    richtig herum sortieren, das wird hier mitgeprüft."""
    dates = pd.to_datetime(list(quarters.keys()))
    df = pd.DataFrame(
        {
            "operatingIncome": [v[0] for v in quarters.values()],
            "depreciationAndAmortization": [v[1] for v in quarters.values()],
        },
        index=dates,
    )
    return df.sort_index(ascending=False)


def test_aapl_ebit_ttm_dec_2024_matches_sum_of_real_10q_quarters(model):
    """Reproduziert die vier echten, aus den früheren (Einzelquartals-)
    Cache-Fixtures entnommenen AAPL-Quartalswerte um den Dezember-2024-
    Stichtag (Q1 FY2025 42.832 Mrd., Q4 FY24 29.591 Mrd., Q3 FY24 25.352
    Mrd., Q2 FY24 27.900 Mrd. USD Operating Income) und prüft die daraus
    resultierende TTM-Summe (125.675 Mrd. USD) - bleibt an echten,
    filingsbasierten Zahlen verankert, ohne einen Live-Netzwerkaufruf oder
    einen neu gebumpten Cache-Fixture-Namen zu brauchen."""
    income_statement = _synthetic_income_statement({
        "2024-12-31": (42_832_000_000, 1_000_000),
        "2024-09-30": (29_591_000_000, 1_000_000),
        "2024-06-30": (25_352_000_000, 1_000_000),
        "2024-03-31": (27_900_000_000, 1_000_000),
    })

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"income_statement": income_statement},
    ):
        df = model.calculate_historical_ebit("AAPL", use_cache=False)

    assert isinstance(df, pd.DataFrame)
    assert "EBIT" in df.columns
    dec_2024_ebit = df.loc["2024-12-31", "EBIT"]
    assert dec_2024_ebit == pytest.approx(125_675_000_000, rel=1e-9)


def _synthetic_quarters(n=8, start_operating_income=1_000_000_000, start_da=100_000_000):
    dates = pd.date_range(end="2024-12-31", periods=n, freq="QE")
    return {
        d.strftime("%Y-%m-%d"): (start_operating_income + i * 50_000_000, start_da + i * 5_000_000)
        for i, d in enumerate(dates)
    }


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "JPM"])
def test_ebitda_equals_ebit_plus_depreciation_and_amortization(model, symbol):
    """EBITDA muss für jede Periode exakt EBIT + Abschreibungen entsprechen —
    bricht sofort, falls die Formel in calculate_historical_ebitda verändert
    wird, ohne dass es beabsichtigt ist. Formel-Identität, unabhängig von der
    konkreten Firma - synthetische Daten reichen aus."""
    income_statement = _synthetic_income_statement(_synthetic_quarters())

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"income_statement": income_statement},
    ):
        df = model.calculate_historical_ebitda(symbol, use_cache=False)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    expected = df["EBIT"] + df["depreciationAndAmortization"]
    pd.testing.assert_series_equal(
        df["EBITDA"], expected, check_names=False, check_exact=True
    )


def test_tangible_book_value_formula_aapl(model):
    """TBV = Total Assets - Intangible Assets - Goodwill - Total Liabilities."""
    df = model.calculate_historical_TangibleBookValue("AAPL")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    expected = (
        df["totalAssets"] - df["intangibleAssets"] - df["goodwill"] - df["totalLiabilities"]
    )
    pd.testing.assert_series_equal(
        df["TangibleBookValue"], expected, check_names=False, check_exact=False, rtol=1e-9
    )


def test_tangible_book_value_formula_msft(model):
    df = model.calculate_historical_TangibleBookValue("MSFT")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    expected = (
        df["totalAssets"] - df["intangibleAssets"] - df["goodwill"] - df["totalLiabilities"]
    )
    pd.testing.assert_series_equal(
        df["TangibleBookValue"], expected, check_names=False, check_exact=False, rtol=1e-9
    )


def test_free_cashflow_formula_aapl(model):
    """FCF = Operating Cashflow - Capital Expenditures."""
    df = model.calculate_historical_FreeCashflow("AAPL")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    expected = df["OperatingCashflow"] - df["CapitalExpenditures"]
    pd.testing.assert_series_equal(
        df["FreeCashflow"], expected, check_names=False, check_exact=True
    )


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "JPM"])
def test_ebit_ebitda_have_no_missing_values(model, symbol):
    """Kein stiller NaN-Ausfall in den Kernspalten — ein Produkt, dessen
    Kernversprechen korrekte Zahlen sind, darf keine unbemerkten Lücken
    durchreichen."""
    income_statement = _synthetic_income_statement(_synthetic_quarters())

    with patch.object(
        model.dataloader, "get_fundamental_data",
        return_value={"income_statement": income_statement},
    ):
        df = model.calculate_historical_ebitda(symbol, use_cache=False)

    assert isinstance(df, pd.DataFrame)
    assert not df["EBIT"].isna().any()
    assert not df["depreciationAndAmortization"].isna().any()
    assert not df["EBITDA"].isna().any()
    assert not df.isin([math.inf, -math.inf]).any().any()


def test_baba_balance_sheet_loads_as_adr(model):
    """ADR-Sonderfall (Alibaba, Foreign Private Issuer): die Bilanz muss auch
    für ausländische Emittenten mit abweichender SEC-Formularstruktur (20-F
    statt 10-K) als valides DataFrame mit den erwarteten Kernpositionen
    laden — kein stiller Absturz oder leeres Ergebnis."""
    balance_sheet = model.dataloader.get_balance_sheet("BABA", frequency="annual")

    assert isinstance(balance_sheet, pd.DataFrame)
    assert not balance_sheet.empty
    assert "Total Assets" in balance_sheet.index
    assert "Total Liabilities Net Minority Interest" in balance_sheet.index
