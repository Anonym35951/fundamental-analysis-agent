"""Regressionstest für einen echten Produktions-Bug (gefunden per Live-Test
durch den Betreiber nach dem EVOLVING.md-Deploy): `run_full_analysis_job()`
rief `resolve_reporting_currency()` auf, ohne es zu importieren (EV-021-
Wiring wurde für `analyze.py` und `custom_analysis.py` korrekt ergänzt, für
`full_analysis.py` aber vergessen). Jede "Vollanalyse" crashte dadurch
SOFORT mit `NameError`, maskiert durch den breiten `except Exception:` in
derselben Funktion als generischer `GENERIC_JOB_ERROR` - "Standard"-Presets
mit Einzelkategorie (z. B. "Wachstumswerte", die über `analyze.py`s
`/{mode}/start` laufen) und "Individuell" (über `custom_analysis.py`) waren
nicht betroffen, weshalb der Fehler bei stichprobenartigem manuellem Testen
unbemerkt blieb.

Kein bestehender Test rief `run_full_analysis_job()` tatsächlich auf - alle
vorhandenen Tests für `/full/start` (`test_analysis_start_symbol_validation.py`,
`test_full_analysis_quota_units_p2_11.py`) mocken `job_manager.submit` weg,
prüfen also nur den synchronen Request-Handler, nie den Job-Body selbst, der
im Hintergrund-Thread läuft. Dieser Test schließt genau diese Lücke, indem
er `run_full_analysis_job()` direkt (synchron, im Testthread) aufruft.
"""
from unittest.mock import MagicMock

import pytest

import api.routes.full_analysis as full_analysis_module
from api.routes.full_analysis import build_analysis_plan, run_full_analysis_job


@pytest.fixture(autouse=True)
def _mock_history_db(monkeypatch):
    # run_full_analysis_job baut sich seine eigene SessionLocal() für die
    # Analyse-Historie - fuer diesen Test irrelevant, daher komplett gemockt
    # statt eine echte Test-DB-Session durchzureichen.
    mock_session = MagicMock()
    monkeypatch.setattr(full_analysis_module, "SessionLocal", lambda: mock_session)
    return mock_session


def _stub_analysis_method(symbol, frequency="annual", use_cache=True):
    # Echte Funktion statt MagicMock: run_full_analysis_job introspiziert
    # `func.__code__.co_varnames`, um zu entscheiden, ob `frequency`
    # mitgegeben wird - ein MagicMock hat kein `__code__`-Attribut.
    return {"ok": True, "symbol": symbol, "frequency": frequency}


def _mock_action():
    action = MagicMock()
    for name in [
        "analyze_wachstumswerte",
        "analyze_dividend_companies",
        "analyze_average_grower",
        "analyze_typical_cyclers",
        "analyze_cycler_turnarounds",
        "analyze_optionality",
        "analyze_asset_play",
    ]:
        setattr(action, name, _stub_analysis_method)
    return action


def test_resolve_reporting_currency_is_importable_in_full_analysis_module():
    """Direkter Nachweis der Ursache: ohne den Import wäre dieser Name im
    Modul-Namespace nicht vorhanden und der Test unten würde mit NameError
    im set_error()-Pfad enden statt mit status=='done'."""
    assert hasattr(full_analysis_module, "resolve_reporting_currency")


def test_run_full_analysis_job_completes_successfully_not_generic_error(monkeypatch):
    action = _mock_action()
    monkeypatch.setattr(full_analysis_module, "get_action", lambda: action)
    monkeypatch.setattr(full_analysis_module, "resolve_reporting_currency", lambda action, symbol: "USD")

    jobs = full_analysis_module.jobs
    plan = build_analysis_plan()
    job_id = jobs.create_job(symbol="AAPL", total=len(plan), user_id=1)

    run_full_analysis_job(job_id, "AAPL")

    result = jobs.get_result(job_id, user_id=None)
    assert result["status"] == "done"
    assert result["error"] is None
    assert result["reporting_currency"] == "USD"
    assert len(result["results"]) == len(plan)


def test_run_full_analysis_job_sets_error_status_when_a_metric_call_fails(monkeypatch):
    """Gegenprobe: ein echter Fehler in der Analyse-Pipeline soll weiterhin
    korrekt als 'error' markiert werden (der breite except-Block bleibt
    ein sinnvolles Sicherheitsnetz - nur die stille Fehlklassifikation durch
    den NameError-Bug war das eigentliche Problem)."""
    action = _mock_action()

    def _failing_analysis_method(symbol, frequency="annual", use_cache=True):
        raise RuntimeError("SEC unavailable")

    action.analyze_wachstumswerte = _failing_analysis_method
    monkeypatch.setattr(full_analysis_module, "get_action", lambda: action)
    monkeypatch.setattr(full_analysis_module, "resolve_reporting_currency", lambda action, symbol: "USD")

    jobs = full_analysis_module.jobs
    plan = build_analysis_plan()
    job_id = jobs.create_job(symbol="AAPL", total=len(plan), user_id=1)

    run_full_analysis_job(job_id, "AAPL")

    result = jobs.get_result(job_id, user_id=None)
    assert result["status"] == "error"
    assert result["error"] == full_analysis_module.GENERIC_JOB_ERROR
