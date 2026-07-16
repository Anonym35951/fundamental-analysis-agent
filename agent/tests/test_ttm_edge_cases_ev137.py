"""EVOLVING.md EV-137: TTM-Edge-Case-Tests.

Da "ttm" für die TTM_CAPABLE_METRICS-Methoden (agent/frequency.py) eine reine
Delegation auf den bestehenden Annual-Codepfad ist (resolve_ttm_alias),
erbt jeder Edge Case, den Annual heute schon korrekt behandelt, dasselbe
Verhalten für "ttm" - ohne neue Numerik, ohne neuen Code für den Edge Case
selbst. Diese Tests beweisen genau das explizit für die in EVOLVING.md §19
geforderten Fälle, per Mock (kein Netzwerk, kein Cache-Zufall).
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from agent.Model import Model


@pytest.fixture
def model() -> Model:
    return Model()


def _quarterly_financials(rows: dict[str, dict[str, float]]) -> pd.DataFrame:
    """rows: {label: {date_str: value}} -> DataFrame mit Labels als Index,
    Perioden (neueste zuerst) als Spalten - wie DataLoader.get_stock_financials
    es zurückgibt."""
    df = pd.DataFrame(rows).T
    df.columns = pd.to_datetime(df.columns)
    return df.sort_index(axis=1, ascending=False)


def test_ttm_with_fewer_than_four_quarters_returns_same_error_as_annual(model: Model):
    """E1 (EVOLVING.md §11): <4 vollständige Quartale -> klare Fehlermeldung,
    kein Hochrechnen, kein stiller Fallback. calculate_kuv's Annual-Zweig
    prueft das bereits selbst; "ttm" muss exakt denselben Fehler erben."""
    financials = _quarterly_financials({
        "Total Revenue": {"2024-03-31": 100.0, "2023-12-31": 90.0, "2023-09-30": 80.0},
    })

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials), \
         patch.object(model.dataloader, "get_current_price_per_share", return_value=150.0), \
         patch.object(model, "_resolve_shares_outstanding", return_value=1_000_000.0):
        annual_result = model.calculate_kuv("TEST", frequency="annual")
        ttm_result = model.calculate_kuv("TEST", frequency="ttm")

    assert "error" in annual_result
    assert "Nicht genügend Quartalsdaten" in annual_result["error"]
    assert ttm_result == annual_result


def test_ttm_with_exactly_four_quarters_computes_successfully(model: Model):
    financials = _quarterly_financials({
        "Total Revenue": {
            "2024-03-31": 100.0, "2023-12-31": 90.0, "2023-09-30": 80.0, "2023-06-30": 70.0,
        },
    })

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials), \
         patch.object(model.dataloader, "get_current_price_per_share", return_value=150.0), \
         patch.object(model, "_resolve_shares_outstanding", return_value=1_000_000.0):
        annual_result = model.calculate_kuv("TEST", frequency="annual")
        ttm_result = model.calculate_kuv("TEST", frequency="ttm")

    assert "error" not in annual_result
    assert annual_result["KUV"] == ttm_result["KUV"]
    assert ttm_result["frequency"] == "ttm"
    assert annual_result["frequency"] == "annual"


def test_ttm_with_negative_net_income_returns_same_result_as_annual(model: Model):
    """Negative Werte: annual behandelt sie bereits (kein Extra-Fall für
    ttm noetig, da reine Delegation)."""
    financials = _quarterly_financials({
        "Net Income": {
            "2024-03-31": -50.0, "2023-12-31": -30.0, "2023-09-30": -20.0, "2023-06-30": -10.0,
        },
    })
    balance_sheet = pd.DataFrame(
        {"2024-03-31": [500.0]}, index=["Stockholders Equity"]
    )
    balance_sheet.columns = pd.to_datetime(balance_sheet.columns)

    with patch.object(model.dataloader, "get_stock_financials", return_value=financials), \
         patch.object(model.dataloader, "get_balance_sheet", return_value=balance_sheet):
        annual_result = model.calculate_roe("TEST", frequency="annual")
        ttm_result = model.calculate_roe("TEST", frequency="ttm")

    assert annual_result["ROE"] == ttm_result["ROE"]
    assert annual_result["ROE"] < 0
    assert ttm_result["frequency"] == "ttm"


def test_ttm_invalid_frequency_guard_still_rejects_garbage_values(model: Model):
    """resolve_ttm_alias uebersetzt ausschliesslich "ttm" - jeder andere
    ungueltige String muss weiterhin den bestehenden Guard-Fehler treffen,
    unveraendert durch die TTM-Aenderung."""
    result = model.calculate_ev_to_sales("TEST", frequency="halbjaehrlich")
    assert "error" in result
    assert "Ungültige Frequenz: halbjaehrlich" in result["error"]
