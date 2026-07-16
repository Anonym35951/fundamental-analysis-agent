import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { startCustomAnalysis, getCustomAnalysisProgress, getCustomAnalysisResult } from "../api/customAnalysis";
import { ApiError } from "../api/client";
import type { MetricSelection } from "../types/customAnalysis";
import { useToast } from "../components/ui/useToast";
import { CompareContext, type CompanyResult, type CompareContextValue, type CompareFrequency } from "./compareContextValue";

export type { CompanyResult, CompareFrequency } from "./compareContextValue";

const STORAGE_KEY = "compare_workspace_v2";
const POLL_INTERVAL_MS = 1500;

type PersistedState = {
  metrics: MetricSelection[];
  companies: CompanyResult[];
  frequency: CompareFrequency;
  hasStarted: boolean;
};

// EV-134: Whitelist statt eines binären "quarterly"-Vergleichs, damit ein
// gültig gespeichertes "ttm" nicht auf "annual" zurückfällt - UND als
// Rollback-Schutz in die andere Richtung: liest ein älterer Build (ohne
// "ttm"-Unterstützung) diesen Wert, würde sein eigener alter Vergleich
// "annual" liefern, was das Backend unverändert akzeptiert (kein Absturz).
const VALID_FREQUENCIES: readonly CompareFrequency[] = ["annual", "quarterly", "ttm"];

function loadFromStorage(): PersistedState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    // EV-135: TTM ist der Default fuer einen frischen Zustand (kein
    // localStorage-Eintrag) - ein bereits gespeicherter Zustand (auch mit
    // "annual") wird unverändert respektiert, siehe VALID_FREQUENCIES-Zweig
    // unten.
    if (!raw) return { metrics: [], companies: [], frequency: "ttm", hasStarted: false };
    const parsed = JSON.parse(raw);
    const storedFrequency = parsed?.frequency;
    return {
      metrics: Array.isArray(parsed?.metrics) ? parsed.metrics : [],
      companies: Array.isArray(parsed?.companies) ? parsed.companies : [],
      frequency: VALID_FREQUENCIES.includes(storedFrequency) ? storedFrequency : "annual",
      hasStarted: Boolean(parsed?.hasStarted),
    };
  } catch {
    return { metrics: [], companies: [], frequency: "ttm", hasStarted: false };
  }
}

function isFullyCovered(company: CompanyResult, selections: MetricSelection[], frequency: CompareFrequency): boolean {
  if (company.frequency !== frequency) return false;
  const covered = Object.keys(company.metrics);
  return selections.every((selection) => {
    if (!covered.includes(selection.key)) return false;
    const lastParams = company.metricParams?.[selection.key] ?? {};
    return JSON.stringify(lastParams) === JSON.stringify(selection.params ?? {});
  });
}

/** Standalone Compare workspace: a shared metric selection applied to any
 * number of companies. The first fetch only happens once `startComparison()`
 * is called (explicit "Vergleich starten" button); after that, adding a
 * company or a metric automatically (re-)runs a custom-analysis job for
 * whichever companies don't yet cover the full current selection; removing a
 * metric is a pure client-side filter (no re-fetch needed, since prior
 * results already hold the data). Persisted to localStorage so the workspace
 * survives reloads. */
export function CompareProvider({ children }: { children: ReactNode }) {
  // Lazy useState-Initializer statt eines Refs (LAUNCH_AUDIT.md P2-10,
  // react-hooks/refs) - loadFromStorage() ist eine reine Lese-Funktion, vier
  // separate (aber je nur einmal beim Mount ausgeführte) Aufrufe sind
  // vernachlässigbar teurer als ein geteilter Ref und brauchen keine
  // Ref-Lesung während des Renders.
  const [metrics, setMetrics] = useState<MetricSelection[]>(() => loadFromStorage().metrics);
  const [companies, setCompanies] = useState<CompanyResult[]>(() => loadFromStorage().companies);
  const [frequency, setFrequency] = useState<CompareFrequency>(() => loadFromStorage().frequency);
  const [hasStarted, setHasStarted] = useState<boolean>(() => loadFromStorage().hasStarted);

  const runningRef = useRef<Set<string>>(new Set());
  const intervalsRef = useRef<Map<string, number>>(new Map());
  const lastStatusRef = useRef<Map<string, CompanyResult["status"]>>(new Map());
  const location = useLocation();
  const navigate = useNavigate();
  const { showToast } = useToast();

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ metrics, companies, frequency, hasStarted }));
  }, [metrics, companies, frequency, hasStarted]);

  useEffect(() => {
    const intervals = intervalsRef.current;
    return () => {
      intervals.forEach((intervalId) => window.clearInterval(intervalId));
      intervals.clear();
    };
  }, []);

  // Beim Mount aus dem (evtl. aus localStorage wiederhergestellten) Stand
  // seeden, damit eine bereits "done"-Firma aus einer früheren Session nicht
  // sofort eine Notification feuert.
  useEffect(() => {
    companies.forEach((c) => lastStatusRef.current.set(c.symbol, c.status));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Status-Übergangs-Watcher: "running" -> "done"/"error" während der Nutzer
  // nicht auf /app/compare ist, löst einen Action-Toast ("Zum Ergebnis") aus
  // — die Inline-Anzeige auf der Compare-Seite selbst reicht dort, kein
  // zusätzliches Polling nötig, da rein auf dem bestehenden companies-State
  // aufgesetzt wird.
  useEffect(() => {
    const onComparePage = location.pathname === "/app/compare";
    companies.forEach((company) => {
      const prev = lastStatusRef.current.get(company.symbol);
      if (prev === "running" && company.status !== "running" && !onComparePage) {
        showToast(
          company.status === "done"
            ? `Vergleich für ${company.symbol} ist fertig.`
            : `Vergleich für ${company.symbol} fehlgeschlagen.`,
          company.status === "done" ? "success" : "error",
          company.status === "done"
            ? { durationMs: 10000, action: { label: "Zum Ergebnis", onClick: () => navigate("/app/compare") } }
            : undefined
        );
      }
      lastStatusRef.current.set(company.symbol, company.status);
    });
  }, [companies, location.pathname, navigate, showToast]);

  const updateCompany = useCallback((symbol: string, patch: Partial<CompanyResult>) => {
    setCompanies((prev) => prev.map((c) => (c.symbol === symbol ? { ...c, ...patch } : c)));
  }, []);

  const runFetch = useCallback(
    (symbol: string, selection: MetricSelection[], freq: CompareFrequency) => {
      if (runningRef.current.has(symbol)) return;
      runningRef.current.add(symbol);
      updateCompany(symbol, { status: "running", error: null });

      const finish = () => {
        runningRef.current.delete(symbol);
        const intervalId = intervalsRef.current.get(symbol);
        if (intervalId !== undefined) {
          window.clearInterval(intervalId);
          intervalsRef.current.delete(symbol);
        }
      };

      startCustomAnalysis({ symbol, metrics: selection, frequency: freq, source: "compare" })
        .then(({ job_id }) => {
          const intervalId = window.setInterval(async () => {
            try {
              const progress = await getCustomAnalysisProgress(job_id);
              if (progress.status === "done") {
                const result = await getCustomAnalysisResult(job_id);
                finish();
                const metricParams: Record<string, Record<string, unknown>> = {};
                for (const sel of selection) {
                  metricParams[sel.key] = sel.params ?? {};
                }
                updateCompany(symbol, {
                  status: "done",
                  metrics: result.metrics,
                  metricParams,
                  frequency: freq,
                  error: null,
                  reporting_currency: result.reporting_currency,
                });
              } else if (progress.status === "error") {
                finish();
                updateCompany(symbol, { status: "error", error: progress.error ?? "Fehler beim Abrufen." });
              }
            } catch (err) {
              finish();
              // Job-ID im JobManager unbekannt (in-memory Job-State) —
              // typischerweise ein Server-Neustart/Deploy während die
              // Analyse lief. Klare Meldung statt generischem Fehler, damit
              // der Nutzer weiß, dass er neu starten muss statt zu warten.
              const isJobLost = err instanceof ApiError && err.status === 404;
              updateCompany(symbol, {
                status: "error",
                error: isJobLost
                  ? "Analyse nicht mehr verfügbar (z. B. durch einen Server-Neustart unterbrochen). Bitte erneut starten."
                  : "Fehler beim Abrufen.",
              });
            }
          }, POLL_INTERVAL_MS);
          intervalsRef.current.set(symbol, intervalId);
        })
        .catch((error: unknown) => {
          finish();
          updateCompany(symbol, {
            status: "error",
            error: error instanceof Error ? error.message : "Analyse konnte nicht gestartet werden.",
          });
        });
    },
    [updateCompany]
  );

  // Coverage-driven (re-)fetch: any company whose last result doesn't cover
  // every currently selected metric key (or was fetched at a different
  // frequency) gets a fresh job, as long as one isn't already in flight.
  // Companies that ended in "error" are intentionally NOT auto-retried here
  // — without this, a persistent failure (e.g. an expired/missing auth
  // token) would re-trigger every render indefinitely, since the failed
  // company is by definition never "covered". The UI's existing recovery
  // path (remove + re-add the company) is the only way to retry those.
  useEffect(() => {
    if (!hasStarted) return;
    if (metrics.length === 0) return;

    companies.forEach((company) => {
      if (runningRef.current.has(company.symbol)) return;
      if (company.status === "error") return;
      if (isFullyCovered(company, metrics, frequency)) return;
      runFetch(company.symbol, metrics, frequency);
    });
  }, [hasStarted, metrics, companies, frequency, runFetch]);

  const startComparison = useCallback(() => {
    setHasStarted(true);
  }, []);

  const addCompany = useCallback((symbol: string) => {
    const cleanSymbol = symbol.trim().toUpperCase();
    if (!cleanSymbol) return;

    setCompanies((prev) => {
      if (prev.some((c) => c.symbol === cleanSymbol)) return prev;
      return [...prev, { symbol: cleanSymbol, status: "running", metrics: {} }];
    });
  }, []);

  const removeCompany = useCallback((symbol: string) => {
    setCompanies((prev) => prev.filter((c) => c.symbol !== symbol));
    const intervalId = intervalsRef.current.get(symbol);
    if (intervalId !== undefined) {
      window.clearInterval(intervalId);
      intervalsRef.current.delete(symbol);
    }
    runningRef.current.delete(symbol);
  }, []);

  const clearAll = useCallback(() => {
    intervalsRef.current.forEach((intervalId) => window.clearInterval(intervalId));
    intervalsRef.current.clear();
    runningRef.current.clear();
    setMetrics([]);
    setCompanies([]);
    setHasStarted(false);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  // Every logout path (explicit button, inactivity timeout, the global 401
  // handler in api/client.ts) dispatches this event instead of touching
  // compare state directly — none of those call sites are guaranteed to be
  // React components with access to this context, so a window event is the
  // one mechanism all of them can reach. Without this, a stale/error'd
  // workspace survives logout and is visible to the next person who opens
  // the browser (or re-triggers the retry loop fixed elsewhere).
  useEffect(() => {
    const handleLogoutEvent = () => clearAll();
    window.addEventListener("app:logout", handleLogoutEvent);
    return () => window.removeEventListener("app:logout", handleLogoutEvent);
  }, [clearAll]);

  const value = useMemo<CompareContextValue>(
    () => ({
      metrics,
      companies,
      frequency,
      hasStarted,
      setMetrics,
      setFrequency,
      addCompany,
      removeCompany,
      startComparison,
      clearAll,
    }),
    [metrics, companies, frequency, hasStarted, addCompany, removeCompany, startComparison, clearAll]
  );

  return <CompareContext.Provider value={value}>{children}</CompareContext.Provider>;
}

