"""Regressionstests für LAUNCH_AUDIT.md P2-5: Model.analyze_dividend_history
schloss das laufende, noch nicht abgeschlossene Kalenderjahr in die
Jahres-CAGR ein - ein Teiljahr-Wert (z.B. nur Januar-Dividende) verzerrte
sowohl "years_with_dividends" als auch den CAGR-Endwert massiv nach unten.
Zusätzlich: calculate_historical_dividend_yield_average nutzte den
deprecated pandas-Resample-Alias 'A' statt 'YE' - reiner Verhaltens-Guard,
dass das Ergebnis nach der Umstellung unverändert bleibt.

Kein Netzwerkzugriff: get_dividend_history/get_stock_data werden gemockt.
Dates sind relativ zu pd.Timestamp.now() konstruiert, damit der Test nicht
mit der Zeit veraltet.
"""
from unittest.mock import patch

import pandas as pd
import pytest


def test_analyze_dividend_history_excludes_current_partial_year(model):
    now = pd.Timestamp.now()
    current_year = now.year
    dates = pd.to_datetime(
        [
            f"{current_year - 3}-06-30",
            f"{current_year - 2}-06-30",
            f"{current_year - 1}-06-30",
            f"{current_year}-01-15",  # laufendes, unvollständiges Jahr
        ]
    )
    # Der Teiljahr-Wert (0.3) ist absichtlich klein - repräsentiert nur eine
    # einzelne bereits gezahlte Dividende im laufenden Jahr, nicht das volle
    # Jahr. Ohne den Fix würde dieser Wert als "letztes Jahr" in die CAGR
    # einfließen und einen massiven künstlichen Einbruch vortäuschen.
    dividends_history = pd.Series([1.0, 1.1, 1.2, 0.3], index=dates, name="dividend")

    with patch.object(
        model.dataloader, "get_dividend_history", return_value={"dividends_history": dividends_history}
    ):
        result = model.analyze_dividend_history("TEST")

    assert "error" not in result
    # Nur die 3 abgeschlossenen Jahre zählen, das laufende Teiljahr nicht.
    assert result["years_with_dividends"] == 3
    assert result["cagr_period_years"] == 2
    # CAGR aus 1.0 -> 1.2 über 2 Jahre (~9.5%), NICHT aus 1.0 -> 0.3 (~-31%),
    # was passieren würde, wenn das Teiljahr mitgezählt würde.
    assert result["cagr"] == pytest.approx(9.54, abs=0.1)


def test_analyze_dividend_history_no_partial_year_unaffected(model):
    """Ohne ein laufendes Teiljahr (letzter Datenpunkt liegt in einem
    abgeschlossenen Vorjahr) darf sich am Ergebnis nichts ändern."""
    now = pd.Timestamp.now()
    current_year = now.year
    dates = pd.to_datetime(
        [f"{current_year - 3}-06-30", f"{current_year - 2}-06-30", f"{current_year - 1}-06-30"]
    )
    dividends_history = pd.Series([1.0, 1.1, 1.2], index=dates, name="dividend")

    with patch.object(
        model.dataloader, "get_dividend_history", return_value={"dividends_history": dividends_history}
    ):
        result = model.analyze_dividend_history("TEST")

    assert "error" not in result
    assert result["years_with_dividends"] == 3
    assert result["cagr_period_years"] == 2
    assert result["cagr"] == pytest.approx(9.54, abs=0.1)


def test_dividend_yield_average_uses_ye_alias_and_still_works(model):
    """Verhaltens-Guard fuer den 'A' -> 'YE'-Alias-Wechsel: gleiche
    Jahres-Bucketing-Semantik, nur ohne DeprecationWarning."""
    dates = pd.to_datetime(["2022-06-30", "2023-06-30", "2024-06-30"])
    dividends = pd.Series([2.0, 2.0, 2.0], index=dates, name="dividend")
    price_dates = pd.date_range("2022-01-01", "2024-12-31", freq="D")
    prices = pd.Series(100.0, index=price_dates, name="Close")

    with patch.object(
        model.dataloader,
        "get_dividend_history",
        return_value={"dividends_history": {"dividend": dividends}},
    ), patch.object(
        model.dataloader, "get_stock_data", return_value={"Close": prices}
    ):
        result = model.calculate_historical_dividend_yield_average("TEST", years=3)

    # 2.0 / 100.0 * 100 = 2% Rendite pro Jahr, gemittelt über 3 Jahre.
    assert result == pytest.approx(2.0, abs=0.01)
