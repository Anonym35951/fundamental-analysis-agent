"""EVOLVING.md EV-132: 'ttm' delegiert für die 15 in
agent/frequency.py::TTM_CAPABLE_METRICS gelisteten Model.py-Methoden 1:1 auf
den bestehenden 'annual'-Pfad. Dieser Test beweist die Alias-Eigenschaft
explizit: `frequency="ttm"` und `frequency="annual"` müssen für jede dieser
Methoden ein bis auf das "frequency"-Label IDENTISCHES Ergebnis liefern -
das ist der von EVOLVING.md geforderte "ttm == annual"-Vertrag.

Determinismus/kein Netzwerk: gleiche Strategie wie
test_golden_master_frequencies.py (siehe dort für die Begründung) -
Cache-Freshness wird ignoriert, der aktuelle Kurs fest verdrahtet.
"""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd
import pytest
import requests
import yfinance as yf

from agent.AgentAction import AgentAction
from agent.DataLoader import DataLoader
from agent.data_sources.sec_source import SecSource
from agent.frequency import TTM_CAPABLE_METRICS

SYMBOL = "AAPL"
FIXED_PRICE = 200.0


def _block_network(*_a: Any, **_kw: Any) -> Any:
    raise AssertionError("test_ttm_delegation_ev132 hat einen echten Netzwerkzugriff ausgelöst.")


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


def _strip_volatile(obj: Any) -> Any:
    """Entfernt/normalisiert Felder, die für den annual==ttm-Vergleich
    irrelevant sind (nur "frequency" darf abweichen)."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: _strip_volatile(v) for k, v in obj.items() if k != "frequency"}
    return obj


@pytest.mark.parametrize("method_name", sorted(TTM_CAPABLE_METRICS))
def test_ttm_delegates_to_identical_annual_result(method_name: str) -> None:
    action = AgentAction(symbol=SYMBOL)
    method = getattr(action.model, method_name)

    annual_result = method(SYMBOL, frequency="annual")
    ttm_result = method(SYMBOL, frequency="ttm")

    assert _strip_volatile(annual_result) == _strip_volatile(ttm_result), (
        f"{method_name}: ttm-Ergebnis weicht (abgesehen vom frequency-Label) vom annual-Ergebnis ab"
    )

    # Das Antwort-Dict muss den ORIGINAL angeforderten Wert zeigen, nicht den
    # intern uebersetzten "annual" - sonst waere die Delegation fuer den
    # Aufrufer unsichtbar/falsch beschriftet.
    if isinstance(ttm_result, dict) and "frequency" in ttm_result:
        assert ttm_result["frequency"] == "ttm"
    if isinstance(annual_result, dict) and "frequency" in annual_result:
        assert annual_result["frequency"] == "annual"


def test_quarterly_is_unaffected_by_ttm_delegation() -> None:
    """Stichprobe: die quarterly-Ergebnisse duerfen durch die neue
    ttm-Uebersetzung nicht veraendert werden (frequency="quarterly" nimmt
    weiterhin den unveraenderten quarterly-Zweig)."""
    action = AgentAction(symbol=SYMBOL)
    for method_name in sorted(TTM_CAPABLE_METRICS):
        method = getattr(action.model, method_name)
        result = method(SYMBOL, frequency="quarterly")
        if isinstance(result, dict) and "frequency" in result:
            assert result["frequency"] == "quarterly", method_name


def test_invalid_frequency_other_than_ttm_is_still_rejected_or_passed_through_unchanged() -> None:
    """resolve_ttm_alias darf ausschließlich "ttm" uebersetzen - jeder andere
    ungueltige Wert muss weiterhin das bisherige (Guard-Fehler oder
    stillschweigendes Else-Verhalten) zeigen, unveraendert durch EV-132."""
    action = AgentAction(symbol=SYMBOL)
    # calculate_ev_to_sales hat einen expliziten Guard -> Fehler-Dict.
    result = action.model.calculate_ev_to_sales(SYMBOL, frequency="banana")
    assert "error" in result
    assert "Ungültige Frequenz: banana" in result["error"]
