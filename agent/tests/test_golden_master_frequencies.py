"""EVOLVING.md EV-101: Golden-Master-Snapshots für den gesamten Metrik-
Katalog unter `frequency=annual` und `frequency=quarterly`.

Zweck: Vor jeder TTM-Änderung (EV-130 ff.) beweisen, dass Annual- und
Quarterly-Ergebnisse byte-identisch bleiben. Ein Diff gegen die eingecheckten
JSON-Dateien in `agent/tests/golden/` ist ein Regressionsalarm.

Determinismus / kein Netzwerk:
- Läuft gegen den echten Produktions-Datei-Cache (`cache/`, Standardpfad von
  `DataLoader()` ohne Override) statt gegen kuratierte Fixtures, weil der
  volle Katalog quer über 4 Symbole realistische SEC-/Alpha-Vantage-Daten
  braucht, die keine schlanke Fixture-Sammlung abdecken kann.
- `cache/` enthält u. a. "price-sensitive" Datentypen mit einer TTL von nur
  10s (agent/DataLoader.py LIVE_CACHE_TTL) bzw. SEC-Ableitungen mit TTLs von
  1-30 Tagen (agent/data_sources/sec_source.py) - jede reale Cache-Datei
  gilt beim Testlauf technisch als "abgelaufen" und würde einen echten
  Netzwerk-Request auslösen. Statt das pro Feld zu mocken (current_price,
  Timestamps, ...), wird hier generisch die Frische-Prüfung selbst
  überschrieben (`ignore_cache_freshness`-Fixture): jede vorhandene
  Cache-Datei gilt für die Dauer des Tests als beliebig frisch. Das liefert
  exakt dieselbe Determinismus-Garantie wie ein Preis-Mock, ohne jede
  volatile Kennzahl einzeln auflisten zu müssen.
- Zusätzliche Absicherung (`no_network`-Fixture): `requests` und
  `yfinance.Ticker.history` sind hart blockiert. Fehlt für ein
  Symbol/eine Metrik ein Cache-Eintrag, schlägt der Aufruf kontrolliert fehl
  (als Exception, die von der Snapshot-Logik als Teil des Golden Masters
  mit aufgezeichnet wird) statt einen echten Request abzusetzen.

Ablauf: Beim ersten Lauf existiert für jedes Symbol noch keine Snapshot-Datei
- sie wird angelegt und der Test übersprungen (mit Hinweis, sie zu
committen). Jeder weitere Lauf vergleicht deterministisch dagegen.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests
import yfinance as yf

from agent.AgentAction import AgentAction
from agent.DataLoader import DataLoader
from agent.data_sources.sec_source import SecSource
from api.services.metric_catalog import METRIC_CATALOG, call_metric

GOLDEN_DIR = Path(__file__).parent / "golden"

# Vorschlag aus EVOLVING.md §17, verifiziert per Cache-Abdeckung
# (ls cache/ | grep '^<SYMBOL>_' | wc -l am 2026-07-16):
#   AAPL 82 Dateien - profitables Large-Cap, hohe Abdeckung
#   AMD  41 Dateien - Zykliker mit saisonalen Quartalen
#   KO   78 Dateien - Dividendenzahler (deckt Dividenden-/Payout-Pfade ab)
#   BABA 81 Dateien - Foreign Private Issuer (20-F), deckt den
#                     Quarterly-Blockade-Pfad ab (sec_source.py:98-106).
# SAP wurde verworfen (nur 23 Cache-Dateien -> zu viele Netzwerk-Lücken).
SYMBOLS: tuple[str, ...] = ("AAPL", "AMD", "KO", "BABA")

FREQUENCIES: tuple[str, ...] = ("annual", "quarterly")

# Feste Fallback-Werte für Date-Params ohne eigenen Default im Katalog, damit
# z. B. calculate_total_inflation_for_period (required=True, kein Default)
# einen echten Snapshot-Wert statt eines "fehlender Parameter"-Fehlers liefert.
_DATE_PARAM_FALLBACKS: dict[str, str] = {
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "current_date_str": "2024-12-31",
    "target_date_str": "2020-12-31",
}


def _block_network(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError(
        "Golden-Master-Test hat einen echten Netzwerkzugriff ausgelöst - "
        "fehlender Cache-Eintrag für dieses Symbol/diese Metrik/Frequenz."
    )


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "get", _block_network)
    monkeypatch.setattr(requests, "post", _block_network)
    monkeypatch.setattr(requests.Session, "get", _block_network)
    monkeypatch.setattr(requests.Session, "post", _block_network)
    monkeypatch.setattr(yf.Ticker, "history", _block_network, raising=False)


_FIXED_CURRENT_PRICES: dict[str, float] = {"AAPL": 200.0, "AMD": 150.0, "KO": 65.0, "BABA": 85.0}


@pytest.fixture(autouse=True)
def fixed_current_price(monkeypatch: pytest.MonkeyPatch) -> None:
    """`DataLoader.get_current_price_per_share` geht NIE über den Datei-Cache
    (siehe DataLoader.py:1078-1095: `_fetch_yahoo_price` wird bei jedem
    Aufruf live abgefragt) - `ignore_cache_freshness` allein reicht hier also
    nicht für Determinismus, weil es keine Cache-Ebene zum "einfrieren" gibt.
    Erstbefund dieses Tasks: ein Testlauf ohne diesen Fix lieferte für
    denselben Code zwei verschiedene Preise (118.4902 vs. 118.5) - ein reales,
    unvorhergesehenes Netzwerk-Leck (yfinance nutzt intern offenbar nicht das
    Standard-`requests`-Modul, sonst hätte `no_network` es gefangen)."""
    monkeypatch.setattr(
        DataLoader, "get_current_price_per_share",
        lambda self, symbol: _FIXED_CURRENT_PRICES.get(symbol.upper(), 100.0),
    )


@pytest.fixture(autouse=True)
def ignore_cache_freshness(monkeypatch: pytest.MonkeyPatch) -> None:
    """Siehe Moduldocstring: macht jede vorhandene Cache-Datei für die Dauer
    des Tests beliebig frisch, statt einzelne volatile Felder zu mocken."""
    forever = pd.Timedelta(days=36500).to_pytimedelta()
    monkeypatch.setattr(DataLoader, "_cache_duration_for", lambda self, data_type: forever)

    original_sec_load = SecSource._load_cached_data

    def _sec_load_ignoring_max_age(self, cache_key, max_age=forever):  # noqa: ANN001
        return original_sec_load(self, cache_key, max_age=forever)

    monkeypatch.setattr(SecSource, "_load_cached_data", _sec_load_ignoring_max_age)


def _default_params(spec, frequency: str) -> dict:
    params: dict[str, Any] = {}
    has_frequency_param = False
    for p in spec.params:
        if p.name == "frequency":
            has_frequency_param = True
            continue
        if p.default is not None:
            params[p.name] = p.default
        elif p.type == "date" and p.name in _DATE_PARAM_FALLBACKS:
            params[p.name] = _DATE_PARAM_FALLBACKS[p.name]
    if has_frequency_param:
        params["frequency"] = frequency
    return params


def _normalize(obj: Any) -> Any:
    """Wandelt ein beliebiges Model.py-Ergebnis (DataFrame, dict, numpy-
    Skalar, NaN/Inf, ...) in eine stabil JSON-serialisierbare Form um."""
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        df.index = [str(ix) for ix in df.index]
        return {"__dataframe__": True, "rows": _normalize(df.to_dict(orient="index"))}
    if isinstance(obj, pd.Series):
        return {"__series__": True, "values": _normalize(obj.to_dict())}
    if isinstance(obj, dict):
        return {str(k): _normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        if math.isnan(f):
            return "NaN"
        if math.isinf(f):
            return "Infinity" if f > 0 else "-Infinity"
        return round(f, 6)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    return str(obj)


def _call_and_normalize(action: AgentAction, spec, symbol: str | None, frequency: str) -> Any:
    params = _default_params(spec, frequency)
    try:
        result = call_metric(action, spec.key, symbol, params)
    except Exception as exc:  # noqa: BLE001 - jeder Fehlermodus gehört zum Golden Master
        result = {"__exception__": type(exc).__name__, "__message__": str(exc)}
    return _normalize(result)


def _assert_matches_snapshot(name: str, data: dict) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.json"
    serialized = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if not path.exists():
        path.write_text(serialized)
        pytest.skip(f"Golden-Snapshot neu erstellt: {path.name} - bitte committen und Test erneut ausführen.")
    expected = path.read_text()
    assert serialized == expected, (
        f"Golden-Master-Abweichung in {path.name} - annual/quarterly-Ergebnisse haben sich "
        f"verändert. Falls beabsichtigt: Snapshot löschen, neu generieren, Diff im PR begründen."
    )


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_golden_master_catalog(symbol: str) -> None:
    action = AgentAction(symbol=symbol)
    snapshot: dict[str, dict[str, Any]] = {}
    for spec in METRIC_CATALOG:
        if not spec.requires_symbol:
            continue
        snapshot[spec.key] = {
            frequency: _call_and_normalize(action, spec, symbol, frequency)
            for frequency in FREQUENCIES
        }
    _assert_matches_snapshot(f"catalog_{symbol}", snapshot)


def test_golden_master_macro_metrics() -> None:
    action = AgentAction()
    snapshot: dict[str, Any] = {}
    for spec in METRIC_CATALOG:
        if spec.requires_symbol:
            continue
        snapshot[spec.key] = _call_and_normalize(action, spec, None, "annual")
    _assert_matches_snapshot("catalog_macro", snapshot)


def test_golden_master_invalid_frequency_error_text() -> None:
    """Fixiert zwei unterschiedliche, real existierende Verhaltensmuster für
    eine ungültige Frequenz - EV-130 muss BEIDE beim Zentralisieren der
    Guards kennen, nicht nur ein einheitliches "if frequency not in [...]"
    annehmen:

    1. `Model.calculate_kuv` hat GAR KEINEN Guard auf Parameter-Ebene: die
       Methode prüft nur `if frequency == "annual"` und behandelt jeden
       anderen String (auch "invalid") wie "quarterly" - liefert also ein
       plausibles, aber für einen Tippfehler falsches Ergebnis, statt eines
       Fehlers. Wichtig für EV-132: ein neuer "ttm"-Wert würde hier OHNE
       Codeänderung STILLSCHWEIGEND als "quarterly" behandelt.
    2. `DataLoader.get_stock_financials` hat den in EVOLVING.md §4.3
       beschriebenen expliziten Guard `if frequency not in ["annual",
       "quarterly"]` und liefert ein Fehler-Dict mit festem Text - dieser
       Text muss nach EV-130 (Zentralisierung) unverändert bleiben.
    """
    action = AgentAction(symbol="AAPL")
    kuv_result = action.model.calculate_kuv("AAPL", frequency="invalid")
    dataloader_result = action.model.dataloader.get_stock_financials("AAPL", frequency="invalid")
    _assert_matches_snapshot(
        "invalid_frequency_error",
        {
            "model_calculate_kuv_no_guard": _normalize(kuv_result),
            "dataloader_get_stock_financials_explicit_guard": _normalize(dataloader_result),
        },
    )


def test_golden_master_foreign_issuer_detection() -> None:
    """Regressionsguard für `SecSource._is_foreign_private_issuer` (Basis des
    20-F-Quarterly-Blockadepfads, sec_source.py:98-106): BABA (20-F/6-K-Filer,
    siehe cache/sec/BABA_companyfacts.json) muss als Foreign Issuer erkannt
    werden, AAPL nicht.

    Hinweis (dokumentierter Ist-Zustand, kein Bug dieses Tasks): ein direkter
    `calculate_debt_to_equity("BABA", frequency="quarterly")`-Aufruf löst den
    Block AKTUELL NICHT aus, weil `SecSource.get_balance_sheet` sein eigenes
    Ergebnis unter `cache/sec/BABA_sec_balance_sheet_quarterly_core.json`
    bereits gecacht hat (offenbar aus einer Zeit vor/ohne diese Prüfung) und
    dieser Cache-Treffer den Foreign-Issuer-Check jedes Mal überspringt, bevor
    er ausgeführt wird. Das ist ein bestehendes, hier nur dokumentiertes
    Verhalten - EV-132 darf sich NICHT darauf verlassen, dass dieser Pfad für
    `ttm` blockiert, ohne das separat zu verifizieren."""
    from agent.data_sources.sec_source import SecSource

    ss = SecSource(user_agent="test@example.com", cache_dir="cache/sec")
    baba_facts = ss.get_company_facts("BABA", use_cache=True)
    aapl_facts = ss.get_company_facts("AAPL", use_cache=True)

    assert ss._is_foreign_private_issuer(baba_facts) is True
    assert ss._is_foreign_private_issuer(aapl_facts) is False
