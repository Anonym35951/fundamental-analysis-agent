import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  getProgress,
  getResult,
  getSingleProgress,
  getSingleResult,
  startFullAnalysis,
  startSingleAnalysisJob,
  type AnalysisMode,
} from "../api/analysis";
import {
  getCustomAnalysisProgress,
  getCustomAnalysisResult,
  runDefinition,
  startCustomAnalysis,
} from "../api/customAnalysis";
import { ApiError } from "../api/client";
import type { CustomAnalysisResult } from "../types/customAnalysis";
import type { FullResult, Progress } from "../types/analysis";
import { useToast } from "../components/ui/useToast";
import { JOB_LOST_MESSAGE } from "../utils/jobErrors";
import {
  AnalysisJobsContext,
  type AnalysisJobKind,
  type AnalyzeJobRecord,
  type AnalysisJobsContextValue,
} from "./analysisJobsContextValue";

const POLL_INTERVAL_MS = 1500;
// Aufräumschwelle für bereits abgeschlossene, benachrichtigte Jobs — deutlich
// konservativer als die 6h-TTL für "done" in api/services/job_manager.py,
// damit das `jobs`-Array über eine lange Sitzung nicht unbegrenzt wächst.
const STALE_JOB_MS = 2 * 60 * 60 * 1000;
const CLEANUP_INTERVAL_MS = 5 * 60 * 1000;

function initialProgress(jobId: string, symbol: string): Progress {
  return {
    job_id: jobId,
    symbol,
    status: "running",
    total: 1,
    done: 0,
    current: "Analyse wird gestartet...",
    error: null,
  };
}

/** Global tracker for Analyze-Seite-Jobs (single/full/custom), analog zum
 * bestehenden CompareProvider: Polling läuft weiter, egal ob der Nutzer auf
 * /app/analyze bleibt. Fertigstellungen außerhalb der Seite lösen einen
 * Action-Toast ("Zum Ergebnis") aus. */
export function AnalysisJobsProvider({ children }: { children: ReactNode }) {
  const [jobs, setJobs] = useState<AnalyzeJobRecord[]>([]);
  const intervalsRef = useRef<Map<string, number>>(new Map());
  const location = useLocation();
  const navigate = useNavigate();
  const { showToast } = useToast();

  useEffect(() => {
    const intervals = intervalsRef.current;
    return () => {
      intervals.forEach((intervalId) => window.clearInterval(intervalId));
      intervals.clear();
    };
  }, []);

  const patchJob = useCallback((id: string, patch: Partial<AnalyzeJobRecord>) => {
    setJobs((prev) => prev.map((job) => (job.id === id ? { ...job, ...patch } : job)));
  }, []);

  const finishPolling = useCallback((id: string) => {
    const intervalId = intervalsRef.current.get(id);
    if (intervalId !== undefined) {
      window.clearInterval(intervalId);
      intervalsRef.current.delete(id);
    }
  }, []);

  const runPoll = useCallback(
    (jobId: string, kind: AnalysisJobKind, mode?: AnalysisMode) => {
      const isFull = kind === "full" || mode === "full";

      const intervalId = window.setInterval(async () => {
        try {
          const nextProgress: Progress =
            kind === "custom"
              ? await getCustomAnalysisProgress(jobId)
              : isFull
                ? await getProgress(jobId)
                : await getSingleProgress(mode as Exclude<AnalysisMode, "full">, jobId);

          patchJob(jobId, { progress: nextProgress });

          if (nextProgress.status === "done") {
            finishPolling(jobId);
            const nextResult: FullResult | CustomAnalysisResult =
              kind === "custom"
                ? await getCustomAnalysisResult(jobId)
                : isFull
                  ? await getResult(jobId)
                  : await getSingleResult(mode as Exclude<AnalysisMode, "full">, jobId);
            patchJob(jobId, { status: "done", result: nextResult, finishedAt: Date.now() });
          } else if (nextProgress.status === "error") {
            finishPolling(jobId);
            patchJob(jobId, {
              status: "error",
              error: nextProgress.error || "Die Analyse konnte nicht abgeschlossen werden.",
              finishedAt: Date.now(),
            });
          }
        } catch (err) {
          finishPolling(jobId);
          // Job-ID im JobManager unbekannt — typischerweise ein Server-
          // Neustart/Deploy während die Analyse lief (in-memory Job-State).
          const isJobLost = err instanceof ApiError && err.status === 404;
          patchJob(jobId, {
            status: "error",
            progress: null,
            error: isJobLost ? JOB_LOST_MESSAGE : "Fehler beim Abrufen des Analysefortschritts.",
            finishedAt: Date.now(),
          });
        }
      }, POLL_INTERVAL_MS);

      intervalsRef.current.set(jobId, intervalId);
    },
    [patchJob, finishPolling]
  );

  const startFullOrSingleJob = useCallback<AnalysisJobsContextValue["startFullOrSingleJob"]>(
    async ({ symbol, mode, frequency, modeLabel }) => {
      const response =
        mode === "full"
          ? await startFullAnalysis(symbol)
          : await startSingleAnalysisJob(symbol, mode as Exclude<AnalysisMode, "full">, frequency);

      const jobId = response?.job_id ?? response?.jobId ?? response?.id ?? null;
      if (!jobId) {
        throw new Error("Keine Job-ID erhalten.");
      }

      const kind: AnalysisJobKind = mode === "full" ? "full" : "single";
      const record: AnalyzeJobRecord = {
        id: jobId,
        kind,
        symbol,
        mode,
        frequency,
        modeLabel,
        definitionId: null,
        definitionName: null,
        status: "running",
        progress: initialProgress(jobId, symbol),
        result: null,
        error: null,
        startedAt: Date.now(),
        finishedAt: null,
        notified: false,
      };
      setJobs((prev) => [...prev, record]);
      runPoll(jobId, kind, mode);
      return jobId;
    },
    [runPoll]
  );

  const startCustomJob = useCallback<AnalysisJobsContextValue["startCustomJob"]>(
    async ({ symbol, metrics, definition }) => {
      const response = definition
        ? await runDefinition(definition.id, { symbol })
        : await startCustomAnalysis({ symbol, metrics: metrics ?? [] });

      const jobId = response.job_id;
      const record: AnalyzeJobRecord = {
        id: jobId,
        kind: "custom",
        symbol,
        modeLabel: definition ? definition.name : "Individuelle Analyse",
        definitionId: definition?.id ?? null,
        definitionName: definition?.name ?? null,
        status: "running",
        progress: initialProgress(jobId, symbol),
        result: null,
        error: null,
        startedAt: Date.now(),
        finishedAt: null,
        notified: false,
      };
      setJobs((prev) => [...prev, record]);
      runPoll(jobId, "custom");
      return jobId;
    },
    [runPoll]
  );

  const getJob = useCallback<AnalysisJobsContextValue["getJob"]>(
    (id) => (id ? jobs.find((job) => job.id === id) : undefined),
    [jobs]
  );

  // Notification-Effect: feuert genau einmal pro Job-Fertigstellung. Auf der
  // Analyze-Seite selbst wird unterdrückt, da die Inline-Anzeige reicht — der
  // "notified"-Flag wird trotzdem gesetzt, damit ein späteres Weg-/Zurück-
  // navigieren keinen nachträglichen Toast auslöst.
  useEffect(() => {
    const onAnalyzePage = location.pathname === "/app/analyze";

    jobs.forEach((job) => {
      if (job.status === "running" || job.notified) return;
      patchJob(job.id, { notified: true });
      if (onAnalyzePage) return;

      if (job.status === "error") {
        showToast(job.error ?? "Die Analyse konnte nicht abgeschlossen werden.", "error");
        return;
      }

      const label = job.modeLabel ?? "Analyse";
      showToast(`${label} für ${job.symbol} ist fertig.`, "success", {
        durationMs: 10000,
        action: {
          label: "Zum Ergebnis",
          onClick: () =>
            navigate("/app/analyze", { state: { viewJob: { jobId: job.id, kind: job.kind } } }),
        },
      });
    });
  }, [jobs, location.pathname, navigate, showToast, patchJob]);

  // Periodisches Aufräumen abgeschlossener, bereits benachrichtigter Jobs.
  useEffect(() => {
    const cleanup = () => {
      const cutoff = Date.now() - STALE_JOB_MS;
      setJobs((prev) =>
        prev.filter((job) => job.status === "running" || (job.finishedAt ?? 0) > cutoff)
      );
    };
    const intervalId = window.setInterval(cleanup, CLEANUP_INTERVAL_MS);
    return () => window.clearInterval(intervalId);
  }, []);

  const value = useMemo<AnalysisJobsContextValue>(
    () => ({ jobs, startFullOrSingleJob, startCustomJob, getJob }),
    [jobs, startFullOrSingleJob, startCustomJob, getJob]
  );

  return <AnalysisJobsContext.Provider value={value}>{children}</AnalysisJobsContext.Provider>;
}
