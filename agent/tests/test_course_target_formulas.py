"""Regressionstests für die in LAUNCH_AUDIT.md dokumentierten Kursziel- und
Inflationsvergleich-Bugs (K-1 bis K-4).

K-1: Price_NetCurrentAssets-Kursziel nutzte Total Assets - Total Liabilities
     (Eigenkapital) statt Total Current Assets - Total Current Liabilities.
K-2: Price_TangibleBookValue-Kursziel unterschlug Goodwill.
K-3: Fehlte das Bilanz-Label "Total Liabilities", fiel der Wert still auf 0
     zurück statt eine tolerante Label-Liste zu prüfen oder einen Fehler zu
     werfen.
K-4: Eine PRO-PERIODE-Wachstumsrate (AAGR/AQGR) wurde direkt mit der
     KUMULATIVEN Inflation über den gesamten Zeitraum verglichen.

Kein Netzwerkzugriff: `model`-Fixture (siehe conftest.py) liest ausschließlich
aus eingefrorenen Cache-Fixtures; die hier zusätzlich benötigten
Bilanz-/Wachstumsdaten werden direkt mit `monkeypatch` auf den DataLoader
bzw. auf Model-Methoden gesetzt, um die Formeln isoliert und deterministisch
zu prüfen.
"""
import math
from unittest.mock import patch

import pandas as pd
import pytest


def _synthetic_multiple_history(column: str) -> pd.DataFrame:
    """Sechs Jahre synthetischer Multiple-Werte, so gewählt, dass
    calculate_buy_case mindestens 3 Jahre VOR dem globalen Tief zur
    Verfügung hat (sonst greift der Fallback-Zweig) und calculate_sell_case
    mindestens 3 verschiedene Jahre insgesamt sieht.

    Werte: 2015=5, 2016=4, 2017=3, 2018=1 (globales Tief), 2019=6, 2020=7.
    Damit ergeben sich reproduzierbare, von Hand nachrechenbare Szenarien:
      BUY  = Median(5, 4, 3) = 4.0
      WC   = round(4.0 / 1.2, 2) = 3.33
      SELL = Median(7, 6, 5) = 6.0
      FV   = (4.0 + 6.0) / 2 = 5.0
    """
    dates = pd.to_datetime(
        ["2015-12-31", "2016-12-31", "2017-12-31", "2018-12-31", "2019-12-31", "2020-12-31"]
    )
    values = [5.0, 4.0, 3.0, 1.0, 6.0, 7.0]
    return pd.DataFrame({column: values}, index=dates)


EXPECTED_WC_MULTIPLE = 3.33
EXPECTED_BUY_MULTIPLE = 4.0
EXPECTED_SELL_MULTIPLE = 6.0
EXPECTED_FV_MULTIPLE = 5.0


def _make_balance_sheet(rows: dict) -> pd.DataFrame:
    """Baut ein echtes pandas DataFrame im get_balance_sheet-Format
    (Zeilenindex = Bilanz-Label, eine Spalte = aktuelle Periode)."""
    return pd.DataFrame({"2024-12-31": rows})


# ---------------------------------------------------------------------------
# K-3: _lookup_balance_sheet_value - tolerante Label-Suche
# ---------------------------------------------------------------------------

def test_lookup_balance_sheet_value_uses_primary_label(model):
    bs = _make_balance_sheet({"Total Liabilities Net Minority Interest": 500.0})
    value = model._lookup_balance_sheet_value(
        bs, ["Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities"]
    )
    assert value == 500.0


def test_lookup_balance_sheet_value_falls_back_to_alternate_label(model):
    """Firmen, deren SEC-Bilanz nur "Total Liabilities" statt "Total
    Liabilities Net Minority Interest" führt, dürfen nicht auf 0 fallen."""
    bs = _make_balance_sheet({"Total Liabilities": 777.0})
    value = model._lookup_balance_sheet_value(
        bs, ["Total Liabilities Net Minority Interest", "Total Liabilities", "TotalLiabilities"]
    )
    assert value == 777.0


def test_lookup_balance_sheet_value_returns_none_when_no_label_matches(model):
    bs = _make_balance_sheet({"Something Else": 1.0})
    value = model._lookup_balance_sheet_value(bs, ["Total Liabilities", "TotalLiabilities"])
    assert value is None


# ---------------------------------------------------------------------------
# K-1: Price_NetCurrentAssets-Kursziel muss Current Assets/Liabilities nutzen
# ---------------------------------------------------------------------------

def test_ncav_course_target_uses_current_assets_and_liabilities_not_equity(model):
    # Total Assets/Liabilities bewusst so gewählt, dass sie ein KOMPLETT
    # anderes (viel größeres) Ergebnis liefern würden als Current
    # Assets/Liabilities - schlägt der Bug (K-1) wieder zu, driftet der Test
    # eindeutig auseinander statt zufällig durchzulaufen.
    balance_sheet = _make_balance_sheet(
        {
            "Total Assets": 100_000.0,
            "Total Liabilities": 80_000.0,
            "Total Current Assets": 1_200.0,
            "Total Current Liabilities": 400.0,
        }
    )
    historical_data = _synthetic_multiple_history("Price_NetCurrentAssets")

    with patch.object(model.dataloader, "get_balance_sheet", return_value=balance_sheet), \
         patch.object(model, "_resolve_shares_outstanding", return_value=100.0):
        result = model.calculate_course_target_PriceMultiples(historical_data, "TEST")

    assert "error" not in result
    # NCAV = 1_200 - 400 = 800; pro Aktie = 800 / 100 = 8.0
    expected_metric_per_share = 8.0
    assert result["buy_price"] == pytest.approx(EXPECTED_BUY_MULTIPLE * expected_metric_per_share, rel=1e-6)
    assert result["sell_price"] == pytest.approx(EXPECTED_SELL_MULTIPLE * expected_metric_per_share, rel=1e-6)
    assert result["fair_value_price"] == pytest.approx(EXPECTED_FV_MULTIPLE * expected_metric_per_share, rel=1e-6)
    assert result["worst_case_price"] == pytest.approx(EXPECTED_WC_MULTIPLE * expected_metric_per_share, rel=1e-6)

    # Regressions-Guard: das (falsche) Eigenkapital-Ergebnis waere um
    # Groessenordnungen hoeher - stellt sicher, dass der Test den Bug
    # tatsaechlich haette auffangen koennen.
    wrong_metric_per_share = (100_000.0 - 80_000.0) / 100.0  # = 200.0
    assert result["buy_price"] != pytest.approx(EXPECTED_BUY_MULTIPLE * wrong_metric_per_share, rel=1e-6)


def test_ncav_course_target_errors_when_current_assets_missing(model):
    """Fehlen Current-Assets/-Liabilities-Labels vollständig, muss ein
    Fehler zurückgegeben werden statt stillschweigend mit 0 zu rechnen."""
    balance_sheet = _make_balance_sheet({"Total Assets": 100_000.0, "Total Liabilities": 80_000.0})
    historical_data = _synthetic_multiple_history("Price_NetCurrentAssets")

    with patch.object(model.dataloader, "get_balance_sheet", return_value=balance_sheet), \
         patch.object(model, "_resolve_shares_outstanding", return_value=100.0):
        result = model.calculate_course_target_PriceMultiples(historical_data, "TEST")

    assert "error" in result


# ---------------------------------------------------------------------------
# K-2 / K-3: Price_TangibleBookValue-Kursziel muss Goodwill abziehen und
# tolerant nach dem Liabilities-Label suchen
# ---------------------------------------------------------------------------

def test_tbv_course_target_subtracts_goodwill(model):
    balance_sheet = _make_balance_sheet(
        {
            "Total Assets": 10_000.0,
            "Intangible Assets": 500.0,
            "Goodwill": 2_000.0,
            # Bewusst NUR die "Net Minority Interest"-Variante, nicht "Total
            # Liabilities" - deckt gleichzeitig die K-3-Label-Toleranz ab.
            "Total Liabilities Net Minority Interest": 3_000.0,
        }
    )
    historical_data = _synthetic_multiple_history("Price_TangibleBookValue")

    with patch.object(model.dataloader, "get_balance_sheet", return_value=balance_sheet), \
         patch.object(model, "_resolve_shares_outstanding", return_value=100.0):
        result = model.calculate_course_target_PriceMultiples(historical_data, "TEST")

    assert "error" not in result
    # TBV = 10_000 - 500 (intangible) - 2_000 (goodwill) - 3_000 (liabilities) = 4_500
    expected_metric_per_share = 4_500.0 / 100.0
    assert result["buy_price"] == pytest.approx(EXPECTED_BUY_MULTIPLE * expected_metric_per_share, rel=1e-6)

    # Regressions-Guard: ohne Goodwill-Abzug (der Bug, K-2) wäre TBV/Aktie
    # um den Goodwill-Anteil zu hoch.
    wrong_metric_per_share = (10_000.0 - 500.0 - 3_000.0) / 100.0
    assert result["buy_price"] != pytest.approx(EXPECTED_BUY_MULTIPLE * wrong_metric_per_share, rel=1e-6)


def test_tbv_course_target_errors_when_total_liabilities_missing(model):
    """Fehlt jede bekannte Liabilities-Variante, darf NICHT still mit 0
    weitergerechnet werden (das frühere Verhalten, K-3)."""
    balance_sheet = _make_balance_sheet(
        {"Total Assets": 10_000.0, "Intangible Assets": 500.0, "Goodwill": 2_000.0}
    )
    historical_data = _synthetic_multiple_history("Price_TangibleBookValue")

    with patch.object(model.dataloader, "get_balance_sheet", return_value=balance_sheet), \
         patch.object(model, "_resolve_shares_outstanding", return_value=100.0):
        result = model.calculate_course_target_PriceMultiples(historical_data, "TEST")

    assert "error" in result


def test_tbv_course_target_treats_missing_goodwill_as_zero_not_error(model):
    """Firmen ganz ohne Goodwill-Position (kein Bilanz-Label) sind ein
    legitimer, haeufiger Fall - kein Fehler, Goodwill einfach 0."""
    balance_sheet = _make_balance_sheet(
        {
            "Total Assets": 10_000.0,
            "Intangible Assets": 0.0,
            "Total Liabilities": 3_000.0,
        }
    )
    historical_data = _synthetic_multiple_history("Price_TangibleBookValue")

    with patch.object(model.dataloader, "get_balance_sheet", return_value=balance_sheet), \
         patch.object(model, "_resolve_shares_outstanding", return_value=100.0):
        result = model.calculate_course_target_PriceMultiples(historical_data, "TEST")

    assert "error" not in result
    expected_metric_per_share = (10_000.0 - 0.0 - 0.0 - 3_000.0) / 100.0
    assert result["buy_price"] == pytest.approx(EXPECTED_BUY_MULTIPLE * expected_metric_per_share, rel=1e-6)


# ---------------------------------------------------------------------------
# K-4: kumuliertes Wachstum vs. kumulative Inflation (statt Rate vs. kumulativ)
# ---------------------------------------------------------------------------

def test_annual_growth_vs_inflation_compares_cumulative_growth(model):
    """3 Jahre a 5%/Jahr kumulieren zu (1.05^3 - 1) = 15.76% - das muss
    gegen eine kumulative Inflation von 12% gewinnen, obwohl die reine
    Ø-Jahresrate (5%) kleiner als 12% ist. Vor dem Fix hätte der alte
    Vergleich (5 > 12 -> False) fälschlich "unterliegt der Inflation"
    gemeldet."""
    growth_result = {
        "avg_growth": 5.0,
        "years": 3,
        "symbol": "TEST",
        "actual_start_date": "2021-12-31",
        "actual_end_date": "2024-12-31",
        "frequency": "annual",
    }
    inflation_result = {"total_inflation": 12.0}

    with patch.object(model, "calculate_avg_annual_profit_growth", return_value=growth_result), \
         patch.object(model, "calculate_total_inflation_for_period", return_value=inflation_result):
        result = model.compare_avg_annual_growth_to_inflation("TEST", None, None)

    assert "error" not in result
    assert result["cumulative_growth"] == pytest.approx(15.76, abs=0.01)
    assert result["outperforms_inflation"] is True


def test_annual_growth_vs_inflation_detects_real_underperformance(model):
    """Gegenprobe: 2% Ø-Jahreswachstum über 3 Jahre kumuliert zu ~6.12% und
    unterliegt einer kumulativen Inflation von 12% - muss weiterhin korrekt
    als "unterliegt" erkannt werden."""
    growth_result = {
        "avg_growth": 2.0,
        "years": 3,
        "symbol": "TEST",
        "actual_start_date": "2021-12-31",
        "actual_end_date": "2024-12-31",
        "frequency": "annual",
    }
    inflation_result = {"total_inflation": 12.0}

    with patch.object(model, "calculate_avg_annual_profit_growth", return_value=growth_result), \
         patch.object(model, "calculate_total_inflation_for_period", return_value=inflation_result):
        result = model.compare_avg_annual_growth_to_inflation("TEST", None, None)

    assert result["cumulative_growth"] == pytest.approx(6.12, abs=0.01)
    assert result["outperforms_inflation"] is False


def test_quarterly_growth_vs_inflation_compares_cumulative_growth(model):
    """8 Quartale a 2%/Quartal kumulieren zu (1.02^8 - 1) = 17.17% und
    schlagen eine kumulative Inflation von 10%, obwohl 2% (Ø pro Quartal)
    kleiner als 10% ist."""
    growth_result = {
        "avg_growth": 2.0,
        "years": 8,
        "symbol": "TEST",
        "actual_start_date": "2023-03-31",
        "actual_end_date": "2025-03-31",
        "frequency": "quarterly",
    }
    inflation_result = {"total_inflation": 10.0}

    with patch.object(model, "calculate_avg_quarterly_profit_growth", return_value=growth_result), \
         patch.object(model, "calculate_total_inflation_for_period", return_value=inflation_result):
        result = model.compare_avg_quarterly_growth_to_inflation("TEST", None, None)

    assert result["cumulative_growth"] == pytest.approx(17.17, abs=0.01)
    assert result["outperforms_inflation"] is True
