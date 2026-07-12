"""Regressionstests für LAUNCH_AUDIT.md P2-1: JobManager._jobs wuchs vorher
unbegrenzt, da nie etwas daraus geloescht wurde. Eviction ist lazy (läuft
bei jedem create_job), daher testen wir sie über direkte Manipulation der
Zeitstempel statt echter Sleeps.

Frische JobManager()-Instanz pro Test statt der geteilten Singleton, damit
Tests sich nicht gegenseitig beeinflussen.
"""
import time

from api.services.job_manager import JobManager


def _make_manager() -> JobManager:
    return JobManager(max_workers=1, max_active_jobs_per_user=3)


def test_finished_job_within_ttl_is_kept():
    jm = _make_manager()
    job_id = jm.create_job(symbol="AAPL", total=1, user_id=1)
    jm.set_done(job_id)

    # Zweiter create_job triggert die Eviction-Sweep - der gerade fertige
    # Job ist weit innerhalb der TTL und darf nicht verschwinden.
    jm.create_job(symbol="MSFT", total=1, user_id=1)

    assert jm.get_progress(job_id, user_id=1) is not None


def test_finished_job_past_ttl_is_evicted():
    jm = _make_manager()
    job_id = jm.create_job(symbol="AAPL", total=1, user_id=1)
    jm.set_done(job_id)
    # Künstlich in die Vergangenheit verschieben, statt Stunden zu schlafen.
    jm._jobs[job_id]["finished_at"] = time.time() - JobManager.FINISHED_JOB_TTL_SECONDS - 1

    jm.create_job(symbol="MSFT", total=1, user_id=1)

    assert jm.get_progress(job_id, user_id=1) is None


def test_error_job_past_ttl_is_evicted():
    jm = _make_manager()
    job_id = jm.create_job(symbol="AAPL", total=1, user_id=1)
    jm.set_error(job_id, "boom")
    jm._jobs[job_id]["finished_at"] = time.time() - JobManager.FINISHED_JOB_TTL_SECONDS - 1

    jm.create_job(symbol="MSFT", total=1, user_id=1)

    assert jm.get_progress(job_id, user_id=1) is None


def test_running_job_within_stuck_timeout_is_kept():
    jm = _make_manager()
    job_id = jm.create_job(symbol="AAPL", total=1, user_id=1)

    jm.create_job(symbol="MSFT", total=1, user_id=1)

    assert jm.get_progress(job_id, user_id=1) is not None


def test_running_job_past_stuck_timeout_is_evicted():
    """Ein Job, der laenger als STUCK_JOB_TTL_SECONDS in "running" haengt,
    kann nur ein verwaister Eintrag sein (Worker-Crash/Deploy) - kein
    legitimer Analyse-Lauf dauert so lange."""
    jm = _make_manager()
    job_id = jm.create_job(symbol="AAPL", total=1, user_id=1)
    jm._jobs[job_id]["started_at"] = time.time() - JobManager.STUCK_JOB_TTL_SECONDS - 1

    jm.create_job(symbol="MSFT", total=1, user_id=1)

    assert jm.get_progress(job_id, user_id=1) is None


def test_eviction_does_not_affect_unrelated_active_job_count():
    """Eviction darf die aktiven-Jobs-Zaehlung fuer noch laufende,
    nicht-abgelaufene Jobs nicht verfaelschen."""
    jm = _make_manager()
    jm.create_job(symbol="AAPL", total=1, user_id=1)
    old_done_id = jm.create_job(symbol="MSFT", total=1, user_id=1)
    jm.set_done(old_done_id)
    jm._jobs[old_done_id]["finished_at"] = time.time() - JobManager.FINISHED_JOB_TTL_SECONDS - 1

    jm.create_job(symbol="GOOGL", total=1, user_id=1)

    assert jm.count_active_jobs(1) == 2
