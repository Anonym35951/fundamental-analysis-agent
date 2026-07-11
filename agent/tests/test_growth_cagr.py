"""Tests für agent/growth_math.compute_net_income_cagr — Fundament der
Methodik-Umstellung von AAGR/AQGR/PEG auf CAGR (LAUNCH.md Block 5,
2026-07-11): arithmetisches Mittel der Einzelperioden-Wachstumsraten
(systematischer Aufwärts-Bias bei volatilen Gewinnen) und QoQ-Vergleich ohne
Saisonbereinigung werden durch eine einzige CAGR-Berechnung aus ältestem und
neuestem positivem Datenpunkt ersetzt.
"""
import math

import pandas as pd
import pytest

from agent.growth_math import compute_net_income_cagr


def _series(dates_values: dict) -> pd.Series:
    return pd.Series(dates_values.values(), index=pd.to_datetime(list(dates_values.keys())))


def test_cagr_two_points_matches_manual_calc():
    # 100 -> 200 über exakt 2 Jahre => CAGR = sqrt(2) - 1 ≈ 41.42%
    s = _series({"2022-01-01": 100.0, "2024-01-01": 200.0})

    result = compute_net_income_cagr(s)

    assert result["years"] == pytest.approx(2.0, abs=0.01)
    assert result["cagr"] == pytest.approx((math.sqrt(2) - 1) * 100, abs=0.1)


def test_cagr_uses_real_day_delta_not_period_count():
    """5 Datenpunkte mit UNGLEICHMÄSSIGEM Abstand (simuliert eine fehlende
    Quartals-Periode) - years muss aus dem tatsächlichen Datumsabstand
    kommen, nicht aus len(net_incomes) - 1."""
    s = _series({
        "2020-01-01": 100.0,
        "2020-04-01": 105.0,
        # Lücke: Q3 2020 fehlt komplett
        "2021-01-01": 110.0,
        "2022-01-01": 121.0,
    })

    result = compute_net_income_cagr(s)

    expected_years = (pd.Timestamp("2022-01-01") - pd.Timestamp("2020-01-01")).days / 365.25
    assert result["years"] == pytest.approx(expected_years, abs=0.01)
    assert result["years"] != pytest.approx(len(s) - 1, abs=0.01)  # nicht positionsbasiert


def test_cagr_walks_inward_when_edges_are_negative():
    """Ältester UND neuester Wert <= 0, aber genug positive Werte dazwischen
    -> nimmt den nächsten validen Rand-Wert statt komplett zu scheitern."""
    s = _series({
        "2020-01-01": -10.0,   # negativer Rand -> übersprungen
        "2021-01-01": 100.0,   # neuer Start
        "2022-01-01": 121.0,   # neues Ende
        "2023-01-01": -5.0,    # negativer Rand -> übersprungen
    })

    result = compute_net_income_cagr(s)

    assert result["start_value"] == 100.0
    assert result["end_value"] == 121.0
    assert result["start_date"] == pd.Timestamp("2021-01-01")
    assert result["end_date"] == pd.Timestamp("2022-01-01")


def test_cagr_errors_when_fewer_than_two_positive_points():
    s = _series({"2020-01-01": -10.0, "2021-01-01": 100.0, "2022-01-01": -5.0})

    result = compute_net_income_cagr(s)

    assert "error" in result


def test_cagr_errors_when_years_delta_is_zero():
    """Zwei Perioden mit identischem Datum (Datenanomalie) - Division durch
    0 im Exponenten muss abgefangen werden, nicht crashen."""
    s = pd.Series([100.0, 200.0], index=pd.to_datetime(["2022-01-01", "2022-01-01"]))

    result = compute_net_income_cagr(s)

    assert "error" in result


def test_cagr_errors_with_fewer_than_two_points_total():
    s = _series({"2022-01-01": 100.0})

    result = compute_net_income_cagr(s)

    assert "error" in result


def test_cagr_ignores_sort_order_of_input():
    """Die Funktion sortiert selbst chronologisch - Eingabe in beliebiger
    Reihenfolge darf das Ergebnis nicht verändern."""
    ascending = _series({"2022-01-01": 100.0, "2024-01-01": 200.0})
    descending = ascending.sort_index(ascending=False)

    result_asc = compute_net_income_cagr(ascending)
    result_desc = compute_net_income_cagr(descending)

    assert result_asc["cagr"] == pytest.approx(result_desc["cagr"])
