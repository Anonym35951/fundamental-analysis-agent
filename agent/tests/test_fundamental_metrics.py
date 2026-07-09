"""Regressionstests für die Kernberechnungen der Analyse-Engine (EBIT,
EBITDA, Free Cashflow, Tangible Book Value).

Datenbasis: eingefrorene, echte SEC-Daten aus dem Produktions-Cache
(agent/tests/fixtures/cache/), nicht synthetisch erzeugt. Kein Netzwerkzugriff
(siehe conftest.py::frozen_cache_dir).

Zwei Arten von Prüfungen:
1. Formel-Identitäten (z. B. EBITDA == EBIT + Abschreibungen) — unabhängig
   von der konkreten Firma nachrechenbar und deckt jede künftige Änderung an
   der Berechnungslogik in Model.py auf.
2. Mindestens ein konkreter, gegen die reale Apple-10-Q-Meldung geprüfter
   Wert (Q1 FY2025 / Dezember-Quartal 2024: Operating Income 42.832 Mrd. USD),
   damit nicht nur die interne Konsistenz, sondern auch die Übereinstimmung
   mit dem tatsächlichen Filing verifiziert ist.
"""
import math

import pandas as pd
import pytest


def test_aapl_ebit_dec_2024_quarter_matches_10q(model):
    """Apples Dezember-2024-Quartal (Q1 FY2025): Operating Income laut
    öffentlichem 10-Q ca. 42,8 Mrd. USD."""
    df = model.calculate_historical_ebit("AAPL")

    assert isinstance(df, pd.DataFrame)
    assert "EBIT" in df.columns

    dec_2024_ebit = df.loc["2024-12-31", "EBIT"]
    assert dec_2024_ebit == pytest.approx(42_832_000_000, rel=1e-9)


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT"])
def test_ebitda_equals_ebit_plus_depreciation_and_amortization(model, symbol):
    """EBITDA muss für jede Periode exakt EBIT + Abschreibungen entsprechen —
    bricht sofort, falls die Formel in calculate_historical_ebitda verändert
    wird, ohne dass es beabsichtigt ist."""
    df = model.calculate_historical_ebitda(symbol)

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


def test_ebitda_formula_holds_for_bank(model):
    """Bank-Sonderfall (JPM): andere Bilanzstruktur als Industrieunternehmen,
    die EBITDA-Formel (EBIT + D&A) muss trotzdem exakt aufgehen."""
    df = model.calculate_historical_ebitda("JPM")

    assert isinstance(df, pd.DataFrame)
    assert not df.empty

    expected = df["EBIT"] + df["depreciationAndAmortization"]
    pd.testing.assert_series_equal(
        df["EBITDA"], expected, check_names=False, check_exact=True
    )


@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "JPM"])
def test_ebit_ebitda_have_no_missing_values(model, symbol):
    """Kein stiller NaN-Ausfall in den Kernspalten — ein Produkt, dessen
    Kernversprechen korrekte Zahlen sind, darf keine unbemerkten Lücken
    durchreichen."""
    df = model.calculate_historical_ebitda(symbol)

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
