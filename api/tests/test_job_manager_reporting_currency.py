"""Regressionstests für EVOLVING.md EV-021: `reporting_currency` muss
additiv durch JobManager laufen - vorhanden (als `None`) direkt nach
create_job, änderbar über set_reporting_currency, und Teil des von
get_result gelieferten Dicts (nicht aber von get_progress, das laut Plan
nur Fortschritt, keine Ergebnis-Metadaten liefert).

Frische JobManager()-Instanz statt des geteilten Singletons, Stil wie
test_job_manager_eviction_p2_1.py.
"""
from api.services.job_manager import JobManager


def _make_manager() -> JobManager:
    return JobManager(max_workers=1, max_active_jobs_per_user=3)


def test_new_job_has_reporting_currency_none_by_default():
    jm = _make_manager()
    job_id = jm.create_job(symbol="AAPL", total=1, user_id=1)

    result = jm.get_result(job_id, user_id=1)

    assert result["reporting_currency"] is None


def test_set_reporting_currency_is_reflected_in_result():
    jm = _make_manager()
    job_id = jm.create_job(symbol="BABA", total=1, user_id=1)

    jm.set_reporting_currency(job_id, "USD")

    assert jm.get_result(job_id, user_id=1)["reporting_currency"] == "USD"


def test_set_reporting_currency_accepts_none_for_undetermined_currency():
    jm = _make_manager()
    job_id = jm.create_job(symbol="SAP", total=1, user_id=1)

    jm.set_reporting_currency(job_id, None)

    assert jm.get_result(job_id, user_id=1)["reporting_currency"] is None


def test_set_reporting_currency_on_unknown_job_id_is_a_noop():
    jm = _make_manager()

    jm.set_reporting_currency("does-not-exist", "USD")  # darf nicht werfen


def test_existing_result_keys_are_unchanged(monkeypatch):
    """Snapshot-Schutz: get_result darf keine bestehenden Keys verlieren
    oder umbenennen, nur additiv erweitern."""
    jm = _make_manager()
    job_id = jm.create_job(symbol="AAPL", total=1, user_id=1)
    jm.add_result(job_id, "metric_a", {"value": 42})
    jm.set_done(job_id)

    result = jm.get_result(job_id, user_id=1)

    assert set(result.keys()) == {
        "job_id", "symbol", "status", "total", "done", "error", "results", "reporting_currency",
    }
    assert result["results"] == {"metric_a": {"value": 42}}


def test_get_progress_does_not_include_reporting_currency():
    """get_progress bleibt bewusst unverändert (nur Fortschritt, keine
    Ergebnis-Metadaten) - reporting_currency gehört laut Plan nur zu
    get_result."""
    jm = _make_manager()
    job_id = jm.create_job(symbol="AAPL", total=1, user_id=1)
    jm.set_reporting_currency(job_id, "USD")

    progress = jm.get_progress(job_id, user_id=1)

    assert "reporting_currency" not in progress
