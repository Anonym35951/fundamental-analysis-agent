import { useEffect, useState } from "react";
import { ApiError } from "../api/client";
import type { Progress } from "../types/analysis";

const JOB_LOST_MESSAGE =
  "Diese Analyse ist nicht mehr verfügbar (z. B. durch einen Server-Neustart unterbrochen). Bitte starte sie erneut.";

/** Extracted from the near-identical polling effects that used to live
 * duplicated across standard and custom analysis job tracking. Polls progress
 * every 1.5s until the job reaches "done" or "error", then fetches the
 * final result once. */
export function useJobPolling<TResult>(
  jobId: string | null,
  getProgress: (jobId: string) => Promise<Progress>,
  getResult: (jobId: string) => Promise<TResult>
) {
  const [progress, setProgress] = useState<Progress | null>(null);
  const [result, setResult] = useState<TResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setProgress(null);
    setResult(null);
    setError(null);

    if (!jobId) return;
    const currentJobId = jobId;

    let isCancelled = false;
    let interval: number | null = null;

    async function tick() {
      try {
        const nextProgress = await getProgress(currentJobId);
        if (isCancelled) return;
        setProgress(nextProgress);

        if (nextProgress.status === "done") {
          if (interval !== null) window.clearInterval(interval);
          const nextResult = await getResult(currentJobId);
          if (isCancelled) return;
          setResult(nextResult);
        }

        if (nextProgress.status === "error") {
          if (interval !== null) window.clearInterval(interval);
          setError(nextProgress.error || "Die Analyse konnte nicht abgeschlossen werden.");
        }
      } catch (err) {
        if (isCancelled) return;
        if (interval !== null) window.clearInterval(interval);
        if (err instanceof ApiError && err.status === 404) {
          // Job-ID im JobManager unbekannt — typischerweise ein Server-
          // Neustart/Deploy während die Analyse lief (in-memory Job-State,
          // siehe api/services/job_manager.py). Progress zurücksetzen statt
          // eine veraltete Fortschrittsanzeige stehen zu lassen.
          setProgress(null);
          setError(JOB_LOST_MESSAGE);
        } else {
          setError("Fehler beim Abrufen des Analysefortschritts.");
        }
      }
    }

    // Poll once immediately instead of waiting for the first 1.5s interval
    // tick — otherwise `progress` stays null right after the job starts,
    // which made the sticky bar flicker/disappear for the first second.
    void tick();
    interval = window.setInterval(tick, 1500);

    return () => {
      isCancelled = true;
      if (interval !== null) window.clearInterval(interval);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  return { progress, result, error };
}
