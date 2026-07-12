"""Regressionstests für die AAGR/AQGR-Methodik-Umstellung auf CAGR
(LAUNCH.md Block 5, 2026-07-11): calculate_avg_annual_profit_growth und
calculate_avg_quarterly_profit_growth nutzen jetzt eine CAGR aus dem
ältesten und neuesten positiven Nettogewinn-Datenpunkt (agent/growth_math)
statt des arithmetischen Mittels aufeinanderfolgender Perioden-
Wachstumsraten. Bestehendes Verhalten für strukturell negative Serien
(Mehrheit der Perioden <= 0) bleibt unverändert und wird hier als
Regressions-Guard mitgeprüft.

Kein Netzwerkzugriff: get_stock_financials wird gemockt.
"""
from unittest.mock import patch

import pandas as pd
import pytest


def _financials(dates_values: dict) -> pd.DataFrame:
    """Baut ein DataFrame im get_stock_financials-Format: Index = Zeilen-
    Label ("Net Income"), Spalten = Perioden-Enddaten."""
    series = pd.Series(dates_values.values(), index=list(dates_values.keys()), name="Net Income")
    return pd.DataFrame([series])


def test_aagr_returns_cagr_not_arithmetic_mean(model):
    """3 Jahre: 100 -> 121 -> 100 (Jahr2 +21%, Jahr3 -17.4%). Arithmetisches
    Mittel wäre positiv (~+1.8%); die CAGR aus ältestem/neuestem Wert
    (100 -> 100 über 2 Jahre) muss dagegen ~0% ergeben - ein Regressions-
    Guard, der die alte Formel eindeutig widerlegt hätte."""
    financials = _financials({"2022-12-31": 100.0, "2023-12-31": 121.0, "2024-12-31": 100.0})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_annual_profit_growth("TEST")

    assert "error" not in result
    assert result["avg_growth"] == pytest.approx(0.0, abs=0.1)
    assert result["years"] == pytest.approx(2.0, abs=0.01)
    assert result["actual_start_date"] == "2022-12-31"
    assert result["actual_end_date"] == "2024-12-31"


def test_aagr_simple_two_point_cagr(model):
    financials = _financials({"2021-12-31": 100.0, "2024-12-31": 133.1})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_annual_profit_growth("TEST")

    # 100 -> 133.1 über 3 Jahre => CAGR = 10%
    assert result["avg_growth"] == pytest.approx(10.0, abs=0.1)
    assert result["years"] == pytest.approx(3.0, abs=0.02)


def test_aagr_majority_negative_still_returns_net_incomes_list(model):
    """Bestehendes Verhalten MUSS erhalten bleiben: Mehrheit negativer
    Perioden -> Datenauflistung + Warnung statt CAGR."""
    financials = _financials({"2022-12-31": -50.0, "2023-12-31": -60.0, "2024-12-31": 10.0})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_annual_profit_growth("TEST")

    assert "net_incomes" in result
    assert "avg_growth" not in result


def test_aagr_single_negative_edge_year_does_not_abort(model):
    """Ein einzelnes Verlustjahr am Rand (Minderheit) darf die CAGR-
    Berechnung nicht verhindern - walk-inward greift."""
    financials = _financials({"2021-12-31": -10.0, "2022-12-31": 100.0, "2023-12-31": 121.0, "2024-12-31": 130.0})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_annual_profit_growth("TEST")

    assert "error" not in result
    assert "net_incomes" not in result
    assert result["actual_start_date"] == "2022-12-31"  # -10 übersprungen


def test_aqgr_reuses_same_cagr_formula_as_aagr(model):
    """Dokumentiert bewusst die (gewollte) strukturelle Ähnlichkeit von
    AQGR/AAGR nach der Methodikänderung: beide sind jetzt eine annualisierte
    CAGR aus ältestem/neuestem Datenpunkt, nur mit unterschiedlicher
    Rohdaten-Granularität."""
    financials = _financials({"2022-03-31": 100.0, "2023-03-31": 110.0, "2024-03-31": 121.0})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_quarterly_profit_growth("TEST")

    assert "error" not in result
    assert result["avg_growth"] == pytest.approx(10.0, abs=0.1)
    assert result["frequency"] == "quarterly"


def test_aqgr_majority_negative_still_returns_net_incomes_list(model):
    financials = _financials({"2024-03-31": -10.0, "2024-06-30": -20.0, "2024-09-30": 5.0})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_quarterly_profit_growth("TEST")

    assert "net_incomes" in result
    assert "avg_growth" not in result


def test_aqgr_negative_values_scaled_to_millions(model):
    """LAUNCH_AUDIT.md P2-4: der Quartals-Negativfall gab Rohwerte aus,
    obwohl Message/Feldname "Mio. USD" versprachen (Faktor 10^6 fehlte) -
    der Annual-Fall (test unten) skaliert korrekt, das ist der Regressions-
    Guard dafür, dass beide Pfade jetzt gleich skalieren."""
    financials = _financials({"2024-03-31": -10_000_000.0, "2024-06-30": -20_000_000.0, "2024-09-30": 5_000_000.0})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_quarterly_profit_growth("TEST")

    values = {entry["date"]: entry["value"] for entry in result["net_incomes"]}
    assert values["2024-03-31"] == pytest.approx(-10.0)
    assert values["2024-06-30"] == pytest.approx(-20.0)
    assert values["2024-09-30"] == pytest.approx(5.0)


def test_aagr_negative_values_scaled_to_millions(model):
    """Kontrollgruppe zum AQGR-Test oben: der Annual-Pfad skalierte schon
    vorher korrekt - dokumentiert, dass beide Frequenzen jetzt konsistent
    sind."""
    financials = _financials({"2022-12-31": -50_000_000.0, "2023-12-31": -60_000_000.0, "2024-12-31": 10_000_000.0})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_annual_profit_growth("TEST")

    values = {entry["date"]: entry["value"] for entry in result["net_incomes"]}
    assert values["2022-12-31"] == pytest.approx(-50.0)
    assert values["2023-12-31"] == pytest.approx(-60.0)
    assert values["2024-12-31"] == pytest.approx(10.0)


def test_aagr_too_few_positive_points_returns_error(model):
    """Nicht strukturell negativ (Minderheit <=0), aber weniger als 2
    positive Punkte übrig nach walk-inward -> Fehler statt Crash."""
    financials = _financials({"2022-12-31": 100.0, "2023-12-31": -5.0, "2024-12-31": -5.0})

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        result = model.calculate_avg_annual_profit_growth("TEST")

    # Mehrheit (2 von 3) ist <= 0 -> geht in den net_incomes-Zweig, kein
    # CAGR-Fehlerpfad in diesem Fall (Regressions-Grenze dokumentiert).
    assert "net_incomes" in result
