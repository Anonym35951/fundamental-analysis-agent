"""EVOLVING.md EV-135: TTM wurde Frontend-Default auf AnalyzePage - das
bedeutet die 5 frequenz-fähigen AgentAction-"Whole-Mode"-Methoden
(analyze_wachstumswerte, analyze_typical_cyclers, analyze_cycler_turnarounds,
analyze_optionality, analyze_asset_play) empfangen jetzt routinemäßig
`frequency="ttm"`, nicht nur über einen expliziten Nutzerklick.

Audit-Ergebnis (siehe agent/frequency.py MODE_ALLOWED_FREQUENCIES-Docstring):
nur `analyze_wachstumswerte` hat eigene, riskante interne Verzweigung
(`if frequency == "annual"` entscheidet AAGR vs. AQGR) und wurde deshalb in
EV-132/135 explizit mit `resolve_ttm_alias` am Funktionsanfang abgesichert.
Die anderen 4 reichen `frequency` nur an bereits TTM-fähige Model.py-
Methoden durch (calculate_roe, calculate_cashflow_margin, ...) - "ttm" fließt
dort bereits sicher durch, ohne Codeänderung. Dieser Test beweist genau das:
alle 5 crashen nicht und liefern für "ttm" das inhaltlich selbe Ergebnis wie
für "annual" (bis auf ein etwaiges "frequency"-Echo-Feld).
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest
import requests
import yfinance as yf

from agent.AgentAction import AgentAction
from agent.DataLoader import DataLoader
from agent.data_sources.sec_source import SecSource

SYMBOL = "AAPL"
FIXED_PRICE = 200.0


def _block_network(*_a: Any, **_kw: Any) -> Any:
    raise AssertionError("test_ttm_mode_methods_ev135 hat einen echten Netzwerkzugriff ausgelöst.")


@pytest.fixture(autouse=True)
def deterministic_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    forever = timedelta(days=36500)
    monkeypatch.setattr(DataLoader, "_cache_duration_for", lambda self, data_type: forever)
    monkeypatch.setattr(DataLoader, "get_current_price_per_share", lambda self, symbol: FIXED_PRICE)

    original_sec_load = SecSource._load_cached_data

    def _sec_load_ignoring_max_age(self, cache_key, max_age=forever):  # noqa: ANN001
        return original_sec_load(self, cache_key, max_age=forever)

    monkeypatch.setattr(SecSource, "_load_cached_data", _sec_load_ignoring_max_age)

    monkeypatch.setattr(requests, "get", _block_network)
    monkeypatch.setattr(requests, "post", _block_network)
    monkeypatch.setattr(requests.Session, "get", _block_network)
    monkeypatch.setattr(requests.Session, "post", _block_network)
    monkeypatch.setattr(yf.Ticker, "history", _block_network, raising=False)


def _strip(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k != "frequency"}
    if isinstance(obj, list):
        return [_strip(v) for v in obj]
    return obj


@pytest.mark.parametrize(
    "method_name",
    [
        "analyze_wachstumswerte",
        "analyze_typical_cyclers",
        "analyze_cycler_turnarounds",
        "analyze_optionality",
        "analyze_asset_play",
    ],
)
def test_mode_method_accepts_ttm_and_matches_annual(method_name: str) -> None:
    action = AgentAction(symbol=SYMBOL)
    method = getattr(action, method_name)

    annual_result = method(SYMBOL, frequency="annual")
    ttm_result = method(SYMBOL, frequency="ttm")

    assert "error" not in ttm_result, f"{method_name}(frequency='ttm') lieferte einen Fehler: {ttm_result}"
    assert _strip(annual_result) == _strip(ttm_result), (
        f"{method_name}: ttm-Ergebnis weicht vom annual-Ergebnis ab (abgesehen von 'frequency')"
    )


def test_analyze_wachstumswerte_ttm_uses_annual_growth_branch_not_quarterly() -> None:
    """Regressionsguard für den konkreten Fund aus EV-135: ohne die
    resolve_ttm_alias-Übersetzung am Funktionsanfang würde "ttm" in
    analyze_wachstumswerte am `if frequency == "annual"`-Check vorbeirutschen
    und AQGR (Quartalswachstum) statt AAGR verwenden - mit einer im Vergleich
    zu einem echten "annual"-Request andere Zahl und Beschriftung
    ("Quartalsweises" statt "Jährliches")."""
    action = AgentAction(symbol=SYMBOL)

    annual_result = action.analyze_wachstumswerte(SYMBOL, frequency="annual")
    ttm_result = action.analyze_wachstumswerte(SYMBOL, frequency="ttm")

    assert annual_result["profit_growth"]["value"] == ttm_result["profit_growth"]["value"]
    assert "Jährliches" in ttm_result["profit_growth"]["message"] or ttm_result["profit_growth"]["message"] == ""
    assert "Quartalsweises" not in ttm_result["profit_growth"]["message"]
