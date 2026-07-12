# api/services/job_manager.py
from __future__ import annotations

import threading
import uuid
import time
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

# Generische Fehlermeldungen für Clients — Exception-Details bleiben im Server-Log.
GENERIC_JOB_ERROR = (
    "Analyse fehlgeschlagen — Datenquelle vorübergehend nicht verfügbar oder "
    "Daten unvollständig. Bitte versuche es später erneut."
)
TOO_MANY_ACTIVE_JOBS_DETAIL = (
    "Zu viele gleichzeitige Analysen. Bitte warte, bis eine laufende Analyse "
    "abgeschlossen ist."
)


def sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_json(v) for v in obj]
    return obj


class JobManager:
    # LAUNCH_AUDIT.md P2-1: _jobs wuchs vorher unbegrenzt (kein Eviction) -
    # abgeschlossene Jobs (done/error) veralten nach ein paar Stunden von
    # selbst (Ergebnisse landen ohnehin dauerhaft in analysis_history), ein
    # "running"-Job, der länger als das Stuck-Timeout offen ist, kann nur
    # ein verwaister Eintrag nach einem Crash/Deploy sein - kein legitimer
    # Analyse-Lauf dauert so lange.
    FINISHED_JOB_TTL_SECONDS = 6 * 3600
    STUCK_JOB_TTL_SECONDS = 2 * 3600

    def __init__(self, max_workers: int = 4, max_active_jobs_per_user: int = 3):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}
        # Gemeinsamer Pool statt unbegrenztem Thread-Spawning pro Request:
        # begrenzt die Gesamtlast, überzählige Jobs warten in der Queue.
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="analysis"
        )
        self.max_active_jobs_per_user = max_active_jobs_per_user

    def _evict_expired_locked(self) -> None:
        """Muss innerhalb von self._lock aufgerufen werden. Räumt abgelaufene
        Jobs aus _jobs - lazy statt per Background-Thread, um keinen
        zusätzlichen Lifecycle (Start/Stop, Thread-Leaks) zu verwalten."""
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if (
                job["status"] in ("done", "error")
                and job["finished_at"] is not None
                and now - job["finished_at"] > self.FINISHED_JOB_TTL_SECONDS
            )
            or (
                job["status"] == "running"
                and now - job["started_at"] > self.STUCK_JOB_TTL_SECONDS
            )
        ]
        for job_id in expired:
            del self._jobs[job_id]

    def count_active_jobs(self, user_id: int) -> int:
        with self._lock:
            return sum(
                1
                for job in self._jobs.values()
                if job.get("user_id") == user_id and job["status"] == "running"
            )

    def submit(self, fn: Callable[[], None]) -> None:
        self._executor.submit(fn)

    def create_job(self, symbol: str, total: int, user_id: int) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._evict_expired_locked()
            self._jobs[job_id] = {
                "job_id": job_id,
                "symbol": symbol,
                "status": "running",
                "total": total,
                "done": 0,
                # Bis der Worker-Thread übernimmt, steht der Job in der Queue.
                "current": "In Warteschlange…",
                "started_at": time.time(),
                "finished_at": None,
                "error": None,
                "results": {},
                "user_id": user_id,
            }
        return job_id

    def set_current(self, job_id: str, current: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["current"] = current

    def add_result(self, job_id: str, key: str, result: dict) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["results"][key] = sanitize_for_json(result)
                self._jobs[job_id]["done"] += 1

    def set_done(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "done"
                self._jobs[job_id]["current"] = None
                self._jobs[job_id]["finished_at"] = time.time()

    def set_error(self, job_id: str, message: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["status"] = "error"
                self._jobs[job_id]["error"] = message
                self._jobs[job_id]["current"] = None
                self._jobs[job_id]["finished_at"] = time.time()

    def get_progress(self, job_id: str, user_id: Optional[int] = None) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            # user_id=None is only ever passed by trusted, same-process callers
            # acting on a job_id they just created themselves (e.g. the
            # background thread's own completion snapshot) — never by a
            # user-facing route. Routes always pass the requesting user's id,
            # and a mismatch is treated identically to "job not found" so a
            # wrong/foreign job_id can't be used to probe for existence.
            if not job or (user_id is not None and job.get("user_id") != user_id):
                return None
            return {
                "job_id": job["job_id"],
                "symbol": job["symbol"],
                "status": job["status"],
                "total": job["total"],
                "done": job["done"],
                "current": job["current"],
                "error": job["error"],
            }

    def get_result(self, job_id: str, user_id: Optional[int] = None) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or (user_id is not None and job.get("user_id") != user_id):
                return None
            return {
                "job_id": job["job_id"],
                "symbol": job["symbol"],
                "status": job["status"],
                "total": job["total"],
                "done": job["done"],
                "error": job["error"],
                "results": job["results"],
            }


# ✅ WICHTIG: Singleton-Instanz
job_manager = JobManager()