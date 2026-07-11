"""Regressionstests für die PEG-Ratio-Vereinheitlichung auf CAGR (LAUNCH.md
Block 5, 2026-07-11): DataLoader.get_peg_ratio (Pfad 1, Standardpfad) nutzte
bisher eine eigenständige Zweipunkt-YoY-Formel ((aktuell-vorjahr)/|vorjahr|),
unabhängig von Model.calculate_avg_annual_profit_growth (AAGR, Pfad-2-
Fallback). Jetzt nutzen beide Pfade dieselbe CAGR-Formel
(agent/growth_math.compute_net_income_cagr) - eine Wachstumsdefinition
für PEG statt zwei.

Kein Netzwerkzugriff: get_current_price_per_share/get_shares_outstanding/
get_stock_financials werden gemockt.
"""
from unittest.mock import patch

import pandas as pd
import pytest

from agent.DataLoader import DataLoader


@pytest.fixture
def loader():
    return DataLoader()


def _financials(dates_values: dict) -> pd.DataFrame:
    series = pd.Series(dates_values.values(), index=list(dates_values.keys()), name="Net Income")
    return pd.DataFrame([series])


def _mock_common(loader, financials, price=100.0, shares=1_000_000.0):
    return (
        patch.object(loader, "get_current_price_per_share", return_value=price),
        patch.object(loader, "get_shares_outstanding", return_value={"shares_outstanding": shares}),
        patch.object(loader, "get_stock_financials", return_value=financials),
    )


def test_get_peg_ratio_uses_cagr_not_yoy_two_point(loader):
    """3 Jahre Net-Income: 100 -> 121 -> 100 (Jahr2 +21%, Jahr3 -17.4% YoY).
    Alte Zweipunkt-Formel hätte (100-121)/121*100 ≈ -17.4% (negatives
    Wachstum) geliefert. Neue CAGR (100 -> 100 über 2 Jahre) = 0% - auch
    negativ/null, aber ein anderer Zahlenwert, der die Formel-Umstellung
    eindeutig belegt."""
    financials = _financials({"2022-12-31": 100.0, "2023-12-31": 121.0, "2024-12-31": 100.0})
    m1, m2, m3 = _mock_common(loader, financials)

    with m1, m2, m3:
        result = loader.get_peg_ratio("TEST", use_cache=False)

    assert result["earnings_growth"] == pytest.approx(0.0, abs=0.1)
    assert result["reason"] == "negative_growth"  # 0% zählt als <= 0


def test_get_peg_ratio_positive_cagr_computes_peg(loader):
    financials = _financials({"2021-12-31": 100.0, "2024-12-31": 133.1})
    m1, m2, m3 = _mock_common(loader, financials, price=100.0, shares=10.0)

    with m1, m2, m3:
        result = loader.get_peg_ratio("TEST", use_cache=False)

    assert "error" not in result
    # 100 -> 133.1 über 3 Jahre => CAGR = 10%
    assert result["earnings_growth"] == pytest.approx(10.0, abs=0.1)
    # EPS = latest_net_income / shares = 133.1 / 10 = 13.31; PE = 100/13.31 ≈ 7.51
    # PEG = PE / growth = 7.51 / 10 ≈ 0.75
    assert result["peg_ratio"] == pytest.approx(0.75, abs=0.05)


def test_get_peg_ratio_negative_growth_still_returns_reason_not_error(loader):
    """Bestehendes Verhalten MUSS erhalten bleiben: negatives/keine
    Wachstum -> {"peg_ratio": None, "reason": "negative_growth"}, kein
    "error"-Feld."""
    financials = _financials({"2021-12-31": 200.0, "2024-12-31": 100.0})
    m1, m2, m3 = _mock_common(loader, financials)

    with m1, m2, m3:
        result = loader.get_peg_ratio("TEST", use_cache=False)

    assert "error" not in result
    assert result["peg_ratio"] is None
    assert result["reason"] == "negative_growth"


def test_calculate_peg_ratio_path1_and_path2_agree_on_growth_rate(model):
    """Pfad 1 (DataLoader.get_peg_ratio) und Pfad 2 (Model.calculate_peg_ratio
    Fallback über calculate_avg_annual_profit_growth) müssen bei identischen
    Net-Income-Daten dieselbe Wachstumsrate liefern - eine Definition statt
    zwei."""
    financials = _financials({"2021-12-31": 100.0, "2024-12-31": 133.1})

    with patch.object(model.dataloader, "get_current_price_per_share", return_value=100.0), \
         patch.object(model.dataloader, "get_shares_outstanding", return_value={"shares_outstanding": 10.0}), \
         patch.object(model.dataloader, "get_stock_financials", return_value=financials):
        path1_result = model.dataloader.get_peg_ratio("TEST", use_cache=False)
        path2_growth = model.calculate_avg_annual_profit_growth("TEST")

    assert path1_result["earnings_growth"] == pytest.approx(path2_growth["avg_growth"], abs=0.01)
