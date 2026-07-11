"""Regressionstests für die in LAUNCH_AUDIT.md/LAUNCH.md dokumentierten P1-3
Randfall-Abbrüche: schuldenfreie Firmen und Nicht-Dividendenzahler brachen die
komplette Dividenden-/Average-Grower-Analyse mit einem generischen Fehler ab,
obwohl "kein Zinsaufwand", "keine Dividende" und "negativer Free Cashflow"
gültige, nur schlechte Messwerte sind - kein Datenproblem.

Kein Netzwerkzugriff: alle DataLoader-/Model-Aufrufe werden mit
unittest.mock.patch.object isoliert (gleiches Muster wie
test_course_target_formulas.py), da die Randfälle (Zinsaufwand=0, fehlende
dividendRate, negativer FCF) nicht zuverlässig über eingefrorene reale
SEC-/yfinance-Fixtures reproduzierbar sind.
"""
from unittest.mock import patch

import pytest

from agent.AgentAction import AgentAction


# ---------------------------------------------------------------------------
# Model.calculate_interest_coverage_ratio: Zinsaufwand=0 -> unendliche
# Deckung statt Fehler; EBIT<=0 -> gültiger (schlechter) Messwert statt Fehler
# ---------------------------------------------------------------------------

def test_interest_coverage_ratio_zero_interest_expense_is_infinite_not_error(model):
    with patch.object(
        model.dataloader, "get_ebit_data",
        return_value={"ebit": 1_000_000.0, "symbol": "DEBTFREE", "date": "2024-12-31"},
    ), patch.object(
        model.dataloader, "get_interest_expense_data",
        return_value={"latest_item": {"abs_value": 0.0}, "symbol": "DEBTFREE"},
    ):
        result = model.calculate_interest_coverage_ratio("DEBTFREE")

    assert "error" not in result
    assert result["interest_coverage_ratio"] == float("inf")


def test_interest_coverage_ratio_negative_ebit_is_valid_not_error(model):
    """EBIT <= 0 ist ein gültiger, schlechter Messwert (negative Deckung) -
    kein Grund, die Berechnung mit einem Fehler abzubrechen."""
    with patch.object(
        model.dataloader, "get_ebit_data",
        return_value={"ebit": -500.0, "symbol": "LOSSCO", "date": "2024-12-31"},
    ), patch.object(
        model.dataloader, "get_interest_expense_data",
        return_value={"latest_item": {"abs_value": 100.0}, "symbol": "LOSSCO"},
    ):
        result = model.calculate_interest_coverage_ratio("LOSSCO")

    assert "error" not in result
    assert result["interest_coverage_ratio"] == -5.0


# ---------------------------------------------------------------------------
# DataLoader.get_dividend_data: fehlendes 'dividendRate' -> 0%-Rendite statt
# Fehler (betrifft Nicht-Dividendenzahler, z. B. reine Wachstumswerte)
# ---------------------------------------------------------------------------

class _FakeTicker:
    def __init__(self, info, dividends):
        self.info = info
        self.dividends = dividends


def test_get_dividend_data_missing_dividend_rate_returns_zero_not_error(model):
    import pandas as pd

    fake_ticker = _FakeTicker(
        info={"regularMarketPrice": 150.0},  # bewusst ohne 'dividendRate'
        dividends=pd.Series(dtype=float),
    )
    with patch("agent.DataLoader.yf.Ticker", return_value=fake_ticker):
        result = model.dataloader.get_dividend_data("NODIV", use_cache=False)

    assert "error" not in result
    assert result["dividend_yield"] == 0
    assert result["dividend_rate"] == 0


# ---------------------------------------------------------------------------
# AgentAction.analyze_dividend_companies: schuldenfreie, dividendenlose Firma
# liefert ein strukturiertes Ergebnis (kein Abbruch)
# ---------------------------------------------------------------------------

@pytest.fixture
def dividend_action():
    return AgentAction()


def test_analyze_dividend_companies_debt_free_non_payer_does_not_abort(dividend_action):
    action = dividend_action
    growth_ok = {
        "aagr": 8.0,
        "cumulative_growth": 20.0,
        "total_inflation": 12.0,
        "outperforms_inflation": True,
        "actual_start_date": "2020-01-01",
        "actual_end_date": "2024-01-01",
    }
    quarterly_ok = {
        "aqgr": 2.0,
        "cumulative_growth": 20.0,
        "total_inflation": 12.0,
        "outperforms_inflation": True,
    }

    with patch.object(
        action.model.dataloader, "get_dividend_data",
        return_value={"dividend_yield": 0, "dividend_rate": 0, "latest_dividend": 0},
    ), patch.object(
        action.model, "compare_avg_annual_growth_to_inflation", return_value=growth_ok,
    ), patch.object(
        action.model, "compare_avg_quarterly_growth_to_inflation", return_value=quarterly_ok,
    ), patch.object(
        action.model, "analyze_payout_ratio",
        return_value={"symbol": "DEBTFREE", "payout_ratio": 0.0, "threshold": 75.0, "status": "ok", "message": "Ausschüttungsquote ≤ 75%."},
    ), patch.object(
        action.model, "calculate_interest_coverage_ratio",
        return_value={"interest_coverage_ratio": float("inf"), "symbol": "DEBTFREE", "frequency": "annual", "date": "2024-12-31"},
    ), patch.object(
        action.model, "calculate_net_debt_to_ebitda", return_value=-1.5,
    ), patch.object(
        action.model, "calculate_ev_to_ebit",
        return_value={"ev_to_ebit": 15.0, "symbol": "DEBTFREE", "frequency": "annual", "date": "2024-12-31"},
    ), patch.object(
        action.model, "analyze_dividend_history",
        return_value={"years_with_dividends": 0, "years_with_increases": 0, "cagr": float("nan"), "cagr_period_years": 0},
    ), patch.object(
        action, "calculate_crv_by_sector_multiples", return_value={"note": "stub"},
    ):
        result = action.analyze_dividend_companies("DEBTFREE")

    # Der Kern der Regression: kein Komplettabbruch trotz "Zinsaufwand faktisch
    # unendlich gedeckt" + "keine Dividende".
    assert "error" not in result
    assert result["interest_coverage_ratio"]["value"] == "inf"
    assert result["interest_coverage_ratio"]["meets_criterion"] is True
    assert result["dividend_yield"]["value"] == 0
    assert result["dividend_yield"]["meets_criterion"] is False
    # Dividendenrendite verfehlt das 5%-Kriterium -> Gesamturteil korrekt
    # negativ, aber als vollständiges, strukturiertes Ergebnis statt Fehler.
    assert result["overall_assessment"] == "Dividend Risky"


# ---------------------------------------------------------------------------
# AgentAction.analyze_average_grower: negativer Free Cashflow ist ein nicht
# erfülltes Kriterium, kein Abbruchgrund
# ---------------------------------------------------------------------------

def test_analyze_average_grower_negative_fcf_marks_criterion_not_met_not_abort():
    import pandas as pd

    action = AgentAction()

    ebit_bandwidth = {
        "ebit": {"Price_EBIT": pd.Series([15.0, 20.0, 25.0])},
        # ebit_ratio: unter historischem Mittel (20) -> erfüllt. Seit der
        # TTM-Konsistenz-Umstellung (2026-07-11, LAUNCH.md Block 5) liest
        # analyze_average_grower das hier statt einen eigenen
        # calculate_price_to_ebit-Aufruf zu machen (siehe agent/AgentAction.py).
        "current": {"ebit_ratio": 10.0},
    }
    tbv_bandwidth = {
        "pb": {"Price_TangibleBookValue": pd.Series([1.5, 2.0, 2.5])},
        "current": {"pb_ratio": 1.0},  # unter Median (2.0) und in [0, 1.5] -> erfüllt
    }
    dividend_data = {"dividend_rate": 1.0, "latest_dividend": 1.0, "dividend_yield": 3.0}  # > 2% -> erfüllt

    with patch.object(
        action.model, "evaluate_ebit_bandwidth", return_value=ebit_bandwidth,
    ), patch.object(
        action.model, "evaluate_tbv_bandwidth", return_value=tbv_bandwidth,
    ), patch.object(
        action.dataloader, "get_dividend_data", return_value=dividend_data,
    ), patch.object(
        action.dataloader, "get_reinvested_profit",
        return_value={"reinvested_profit": 100.0},
    ), patch.object(
        action.dataloader, "get_company_profits",
        return_value={"latest_net_income": 500.0},  # 100 > 0.1*500 -> erfüllt
    ), patch.object(
        action.dataloader, "get_free_cashflow",
        return_value={"free_cashflow": -1000.0},  # <-- der Randfall
    ), patch.object(
        action.dataloader, "get_market_cap",
        return_value={"market_cap": 10_000.0},
    ), patch.object(
        action, "calculate_crv_by_sector_multiples", return_value={"note": "stub"},
    ):
        result = action.analyze_average_grower("SHRINKINGCO")

    assert "error" not in result
    assert result["free_cashflow"]["value"] == -1000.0
    assert result["free_cashflow"]["meets_criterion"] is False
    # Nicht "Nicht analysierbar" (fail()), sondern ein vollständiges Urteil.
    assert result["overall_assessment"] == "Not an Average Grower"
