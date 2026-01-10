# api/services/job_manager.py
from __future__ import annotations

import threading
import uuid
import time
import math
from typing import Any, Dict, Optional


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
    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self, symbol: str, total: int) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self._jobs[job_id] = {
                "job_id": job_id,
                "symbol": symbol,
                "status": "running",
                "total": total,
                "done": 0,
                "current": None,
                "started_at": time.time(),
                "finished_at": None,
                "error": None,
                "results": {},
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

    def get_progress(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
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

    def get_result(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
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