import { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate, useSearchParams } from "react-router-dom";
import AnalyzeStickyBar from "../../components/analysis/AnalyzeStickyBar";
import AnalyzeWorkspace, { type AnalysisTab } from "../../components/analysis/AnalyzeWorkspace";
import type { AnalysisMode } from "../../api/analysis";
import { useSymbolSearch } from "../../hooks/useSymbolSearch";
import { addFavorite, getFavorites, removeFavorite } from "../../api/favorites";
import { ApiError } from "../../api/client";
import type { CustomAnalysisDefinition, CustomAnalysisResult, MetricSelection } from "../../types/customAnalysis";
import type { FullResult } from "../../types/analysis";
import AnalyzeResultsDashboard from "../../components/analysis/AnalyzeResultsDashboard";
import { theme } from "../../components/ui/theme";
import { Button, Modal, useToast } from "../../components/ui";
import QuotaExceededModal from "../../components/analysis/QuotaExceededModal";
import DefinitionBuilder from "../../components/customAnalysis/DefinitionBuilder";
import AdHocAnalysisPanel from "../../components/customAnalysis/AdHocAnalysisPanel";
import CustomAnalysisResultsList from "../../components/customAnalysis/CustomAnalysisResultsList";
import LivePriceBadge from "../../components/shared/LivePriceBadge";
import SourceBadge from "../../components/shared/SourceBadge";
import { useCustomAnalysisDefinitions } from "../../hooks/useCustomAnalysisDefinitions";
import { useAnalysisJobs } from "../../hooks/useAnalysisJobsContext";
import type { AnalysisJobKind } from "../../hooks/analysisJobsContextValue";
import { useFirstRunOnboarding } from "../../hooks/useFirstRunOnboarding";
import { useTourStatus } from "../../hooks/useTourStatus";
import { X } from "lucide-react";

type BuilderModalMode = "create" | "edit" | "adhoc" | null;

/** Passed via navigate(..., { state: { rerun } }) from the dashboard history
 * modal's "Nochmal analysieren" action, so AnalyzePage can replay a past run
 * without the caller needing to know which job-start endpoint applies. */
export type RerunPayload = {
  mode: string;
  symbol: string;
  frequency?: string | null;
  definitionId?: number | null;
  metrics?: MetricSelection[];
};

/** Passed via navigate(..., { state: { viewJob } }) from a job-finished Toast's
 * "Zum Ergebnis" action — the job is already tracked by AnalysisJobsProvider,
 * so this only needs to point at it, not carry its data. */
export type ViewJobPayload = {
  jobId: string;
  kind: AnalysisJobKind;
};

const modeOptions: Array<{
  value: AnalysisMode;
  label: string;
  description: string;
}> = [
  {
    value: "full",
    label: "Vollanalyse",
    description: "Alle Analysen in einem Lauf",
  },
  {
    value: "wachstumswerte",
    label: "Wachstumswerte",
    description: "Fokus auf Wachstum und Qualität",
  },
  {
    value: "dividendenwerte",
    label: "Dividendenwerte",
    description: "Fokus auf Dividende und Stabilität",
  },
  {
    value: "average-grower",
    label: "Average Grower",
    description: "Solides Wachstum über längere Zeit",
  },
  {
    value: "typische-zykliker",
    label: "Typische Zykliker",
    description: "Zyklische Unternehmen analysieren",
  },
  {
    value: "turnarounds",
    label: "Turnarounds",
    description: "Sondersituationen und Erholung",
  },
  {
    value: "optionality",
    label: "Optionalitäten",
    description: "Zusätzliche Chancen und Hebel",
  },
  {
    value: "asset-play",
    label: "Asset Play",
    description: "Substanz- und Vermögenswerte",
  },
];

function AnalyzePage() {
  const { showToast } = useToast();
  const [searchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [pendingRerun, setPendingRerun] = useState<RerunPayload | null>(
    (location.state as { rerun?: RerunPayload } | null)?.rerun ?? null
  );
  const [pendingViewJob, setPendingViewJob] = useState<ViewJobPayload | null>(
    (location.state as { viewJob?: ViewJobPayload } | null)?.viewJob ?? null
  );
  const [symbol, setSymbol] = useState("");
  const [selectedMode, setSelectedMode] = useState<AnalysisMode>("full");
  const [selectedFrequency, setSelectedFrequency] = useState<"annual" | "quarterly">("annual");
  const [favoriteSymbols, setFavoriteSymbols] = useState<Set<string>>(new Set());

  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("standard");

  // Onboarding-Tour: der "Eigene Analyse erstellen"-Schritt zielt auf ein
  // Element, das nur im "Individuell"-Tab existiert - erzwingt den
  // Tab-Wechsel, solange dieser Schritt aktiv ist (siehe tourSteps.ts).
  const { currentStepData } = useTourStatus();
  useEffect(() => {
    if (currentStepData?.requiredTab === "individuell" && analysisTab !== "individuell") {
      setAnalysisTab("individuell");
    }
  }, [currentStepData, analysisTab]);

  // Analyse-Job-Tracking lebt global (AnalysisJobsProvider), damit Polling
  // beim Wegnavigieren weiterläuft und eine Fertig-Notification über die
  // ganze App hinweg feuern kann — hier nur die Job-ID-Pointer + davon
  // abgeleiteter Fortschritt/Ergebnis.
  const { startFullOrSingleJob, startCustomJob, getJob } = useAnalysisJobs();

  // Standard-Analyse-Job-State
  const [currentStandardJobId, setCurrentStandardJobId] = useState<string | null>(null);
  const standardJob = getJob(currentStandardJobId);
  const progress = standardJob?.progress ?? null;
  const result = (standardJob?.result as FullResult | null) ?? null;
  const standardJobError = standardJob?.status === "error" ? standardJob.error : null;

  // Individuell: gespeicherte Definitionen + Katalog + CRUD
  const {
    catalog,
    isLoadingCatalog,
    definitions,
    isLoadingDefinitions,
    canCreateNew,
    isAnalysisLimitReached,
    refreshUsage,
    saveDefinition,
    removeDefinition,
    reloadDefinitions,
  } = useCustomAnalysisDefinitions();

  const [selectedDefinition, setSelectedDefinition] = useState<CustomAnalysisDefinition | null>(null);
  const [adHocMetrics, setAdHocMetrics] = useState<MetricSelection[] | null>(null);
  const [builderModalMode, setBuilderModalMode] = useState<BuilderModalMode>(null);
  const [editingDefinition, setEditingDefinition] = useState<CustomAnalysisDefinition | null>(null);
  const [isSavingDefinition, setIsSavingDefinition] = useState(false);

  // Free-Plan-Kontingent aufgebraucht: statt eines toten, deaktivierten
  // Buttons oder einer rohen Fehlermeldung zeigen wir hier den einzigen
  // Moment, in dem ein Free-User Pro tatsächlich braucht, als Einladung statt
  // Frustration. resetDate kommt vom Backend (QUOTA_EXCEEDED-Detail); im
  // präventiven Fall (Klick, bevor der Request überhaupt rausgeht) bleibt sie
  // leer und das Modal zeigt eine generische "zum Monatsanfang"-Formulierung.
  const [quotaModalResetDate, setQuotaModalResetDate] = useState<string | null>(null);
  const [isQuotaModalOpen, setIsQuotaModalOpen] = useState(false);

  function openQuotaModal(resetDate?: string | null) {
    setQuotaModalResetDate(resetDate ?? null);
    setIsQuotaModalOpen(true);
  }

  // Individuell: Job-State (eigener Ad-hoc/Definition-Lauf statt Standard-Job)
  const [currentCustomJobId, setCurrentCustomJobId] = useState<string | null>(null);
  const customJob = getJob(currentCustomJobId);
  const customProgress = customJob?.progress ?? null;
  const customResult = (customJob?.result as CustomAnalysisResult | null) ?? null;
  const customJobError = customJob?.status === "error" ? customJob.error : null;

  const [isStartingAnalysis, setIsStartingAnalysis] = useState(false);
  const [pageErrorMessage, setPageErrorMessage] = useState("");
  const [pageSuccessMessage, setPageSuccessMessage] = useState("");
  const [isSuggestionsOpen, setIsSuggestionsOpen] = useState(false);
  const { isFirstRun, dismissOnboarding } = useFirstRunOnboarding();
  const blurTimeoutRef = useRef<number | null>(null);
  const resultsSectionRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    getFavorites()
      .then((favs) => setFavoriteSymbols(new Set(favs.map((fav) => fav.symbol))))
      .catch(() => setFavoriteSymbols(new Set()));
  }, []);

  // Lets a sidebar favorite click land here with the symbol pre-filled.
  useEffect(() => {
    const symbolFromQuery = searchParams.get("symbol");
    if (symbolFromQuery) {
      setSymbol(symbolFromQuery.toUpperCase());
    }
  }, [searchParams]);

  // Backwards-compatible deep link: the old /app/custom-analysis?run=<id>
  // route now redirects to /app/analyze?run=<id> — land on the Individuell
  // tab with that definition pre-selected instead of duplicating a runner.
  useEffect(() => {
    const runId = searchParams.get("run");
    if (!runId || isLoadingDefinitions) return;

    const definition = definitions.find((entry) => entry.id === Number(runId));
    if (definition) {
      setAnalysisTab("individuell");
      setSelectedDefinition(definition);
      setAdHocMetrics(null);
    }
     
  }, [searchParams, isLoadingDefinitions, definitions]);

  function handleSelectCustomDefinition(definition: CustomAnalysisDefinition) {
    setSelectedDefinition(definition);
    setAdHocMetrics(null);
  }

  function handleRunAdHoc() {
    setBuilderModalMode("adhoc");
  }

  function handleCreateNewDefinition() {
    setEditingDefinition(null);
    setBuilderModalMode("create");
  }

  function handleEditDefinition(definition: CustomAnalysisDefinition) {
    setEditingDefinition(definition);
    setBuilderModalMode("edit");
  }

  async function handleDeleteDefinition(id: number) {
    await removeDefinition(id);
    if (selectedDefinition?.id === id) {
      setSelectedDefinition(null);
    }
  }

  function closeBuilderModal() {
    setBuilderModalMode(null);
    setEditingDefinition(null);
  }

  async function handleBuilderSave(name: string, metrics: MetricSelection[]) {
    try {
      setIsSavingDefinition(true);
      const editingId = builderModalMode === "edit" ? editingDefinition?.id ?? null : null;
      await saveDefinition(name, metrics, editingId);
      closeBuilderModal();
    } catch (error) {
      setPageErrorMessage(
        error instanceof Error ? error.message : "Analyse konnte nicht gespeichert werden."
      );
    } finally {
      setIsSavingDefinition(false);
    }
  }

  // Spiegelt den vom AnalysisJobsProvider getrackten Job-Status auf die
  // gemeinsamen Banner-States, damit Standard- und Individuell-Läufe
  // dieselbe Fehler-/Erfolgs-Anzeige nutzen.
  useEffect(() => {
    if (standardJobError) {
      setPageErrorMessage(standardJobError);
    }
  }, [standardJobError]);

  useEffect(() => {
    if (result) {
      setPageSuccessMessage("Analyse erfolgreich abgeschlossen.");
    }
  }, [result]);

  useEffect(() => {
    if (customJobError) {
      setPageErrorMessage(customJobError);
    }
  }, [customJobError]);

  useEffect(() => {
    if (customResult) {
      setPageSuccessMessage("Analyse erfolgreich abgeschlossen.");
    }
  }, [customResult]);

  // Auto-scroll to the results section once either analysis type finishes —
  // the sticky bar already shows live progress, the user doesn't need to
  // track raw job state manually while waiting.
  useEffect(() => {
    if (result || customResult) {
      resultsSectionRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [result, customResult]);

  // Drop the rerun/viewJob payload from router state once read so a refresh
  // or back-navigation to this page doesn't replay/re-jump again.
  useEffect(() => {
    const state = location.state as { rerun?: RerunPayload; viewJob?: ViewJobPayload } | null;
    if (state && (state.rerun || state.viewJob)) {
      navigate(`${location.pathname}${location.search}`, { replace: true, state: null });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // "Zum Ergebnis" aus einer Job-fertig-Notification: der Job ist bereits im
  // AnalysisJobsProvider getrackt, hier nur Tab + aktive Job-ID setzen, damit
  // das Ergebnis sofort aus dem Cache gerendert wird (kein erneuter Request).
  useEffect(() => {
    if (!pendingViewJob) return;

    const job = getJob(pendingViewJob.jobId);
    if (!job) {
      setPendingViewJob(null);
      return;
    }

    setSymbol(job.symbol);
    if (job.kind === "custom") {
      setAnalysisTab("individuell");
      setCurrentCustomJobId(job.id);
    } else {
      setAnalysisTab("standard");
      if (job.mode) setSelectedMode(job.mode);
      if (job.frequency) setSelectedFrequency(job.frequency);
      setCurrentStandardJobId(job.id);
    }
    setPendingViewJob(null);
  }, [pendingViewJob, getJob]);

  // "Nochmal analysieren" from the dashboard history: replay a past run via
  // the same start endpoints a manual click would use, once any data the
  // replay depends on (saved definitions) has loaded.
  useEffect(() => {
    if (!pendingRerun) return;

    const cleanSymbol = pendingRerun.symbol.trim().toUpperCase();
    setSymbol(cleanSymbol);

    if (pendingRerun.mode !== "custom") {
      const mode = pendingRerun.mode as AnalysisMode;
      const frequency = (pendingRerun.frequency as "annual" | "quarterly" | null) ?? "annual";
      setAnalysisTab("standard");
      setSelectedMode(mode);
      setSelectedFrequency(frequency);
      setPendingRerun(null);
      void handleStartAnalysis({ symbol: cleanSymbol, mode, frequency });
      return;
    }

    if (pendingRerun.definitionId != null) {
      if (isLoadingDefinitions) return;
      const definition = definitions.find((entry) => entry.id === pendingRerun.definitionId) ?? null;
      setAnalysisTab("individuell");
      setPendingRerun(null);
      if (definition) {
        setSelectedDefinition(definition);
        setAdHocMetrics(null);
        void runCustomAnalysis({ definition, symbol: cleanSymbol });
      } else {
        setPageErrorMessage("Die ursprüngliche Analyse-Definition existiert nicht mehr.");
      }
      return;
    }

    if (pendingRerun.metrics) {
      setAnalysisTab("individuell");
      setSelectedDefinition(null);
      setAdHocMetrics(pendingRerun.metrics);
      setPendingRerun(null);
      void runCustomAnalysis({ metrics: pendingRerun.metrics, symbol: cleanSymbol });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingRerun, isLoadingDefinitions, definitions]);

  const normalizedSymbol = symbol.trim().toUpperCase();

  const { suggestions: filteredSuggestions, isLoadingSuggestions: isLoadingSymbols } =
    useSymbolSearch(normalizedSymbol);

  const isFavorited = favoriteSymbols.has(normalizedSymbol);

  async function handleToggleFavorite() {
    if (!normalizedSymbol) return;
    const previous = favoriteSymbols;
    const next = new Set(previous);

    if (isFavorited) {
      next.delete(normalizedSymbol);
      setFavoriteSymbols(next);
      try {
        await removeFavorite(normalizedSymbol);
      } catch {
        setFavoriteSymbols(previous);
        showToast("Konnte nicht gespeichert werden.", "error");
      }
    } else {
      next.add(normalizedSymbol);
      setFavoriteSymbols(next);
      try {
        await addFavorite(normalizedSymbol);
      } catch {
        setFavoriteSymbols(previous);
        showToast("Konnte nicht gespeichert werden.", "error");
      }
    }
  }

  async function handleStartAnalysis(overrides?: {
    symbol?: string;
    mode?: AnalysisMode;
    frequency?: "annual" | "quarterly";
  }) {
    if (isStartingAnalysis) {
      return;
    }
    if (isAnalysisLimitReached) {
      openQuotaModal();
      return;
    }

    setPageErrorMessage("");
    setPageSuccessMessage("");

    const cleanSymbol = overrides?.symbol ?? normalizedSymbol;
    const modeToUse = overrides?.mode ?? selectedMode;
    const frequencyToUse = overrides?.frequency ?? selectedFrequency;

    if (!cleanSymbol) {
      setPageErrorMessage("Bitte gib ein Symbol ein.");
      return;
    }

    try {
      setIsStartingAnalysis(true);
      setCurrentStandardJobId(null);

      const modeLabel = modeOptions.find((option) => option.value === modeToUse)?.label ?? "Analyse";
      const nextJobId = await startFullOrSingleJob({
        symbol: cleanSymbol,
        mode: modeToUse,
        frequency: frequencyToUse,
        modeLabel,
      });

      setCurrentStandardJobId(nextJobId);
      refreshUsage();
    } catch (error) {
      if (error instanceof ApiError && error.code === "QUOTA_EXCEEDED") {
        openQuotaModal(typeof error.data?.reset_date === "string" ? error.data.reset_date : null);
      } else if (error instanceof Error) {
        setPageErrorMessage(error.message);
      } else {
        setPageErrorMessage("Analyse konnte nicht gestartet werden.");
      }
    } finally {
      setIsStartingAnalysis(false);
    }
  }

  // Shared by the "Individuell" tab's main start button (selectedDefinition
  // or a previously built adHocMetrics selection) and the Ad-hoc-Analyse
  // modal (which passes its just-picked metrics directly instead of relying
  // on state set elsewhere).
  async function runCustomAnalysis(options: {
    definition?: CustomAnalysisDefinition | null;
    metrics?: MetricSelection[];
    symbol?: string;
  }) {
    if (isStartingAnalysis) {
      return;
    }
    if (isAnalysisLimitReached) {
      openQuotaModal();
      return;
    }

    setPageErrorMessage("");
    setPageSuccessMessage("");

    const cleanSymbol = options.symbol ?? normalizedSymbol;

    if (!cleanSymbol) {
      setPageErrorMessage("Bitte gib ein Symbol ein.");
      return;
    }

    try {
      setIsStartingAnalysis(true);
      setCurrentCustomJobId(null);

      const nextJobId = await startCustomJob({
        symbol: cleanSymbol,
        metrics: options.metrics,
        definition: options.definition,
      });

      setAnalysisTab("individuell");
      if (options.definition) {
        setSelectedDefinition(options.definition);
        setAdHocMetrics(null);
        // Refresh so the card's "last_run_at" timestamp stays current.
        reloadDefinitions();
      } else {
        setAdHocMetrics(options.metrics ?? null);
        setSelectedDefinition(null);
      }

      setCurrentCustomJobId(nextJobId);
      refreshUsage();
    } catch (error) {
      if (error instanceof ApiError && error.code === "QUOTA_EXCEEDED") {
        openQuotaModal(typeof error.data?.reset_date === "string" ? error.data.reset_date : null);
      } else {
        setPageErrorMessage(
          error instanceof Error ? error.message : "Analyse konnte nicht gestartet werden."
        );
      }
    } finally {
      setIsStartingAnalysis(false);
    }
  }

  function handleStartCustomAnalysis() {
    if (!selectedDefinition && !adHocMetrics) {
      setPageErrorMessage(
        "Bitte wähle eine eigene Analyse aus oder erstelle eine einmalige Analyse."
      );
      return;
    }
    runCustomAnalysis({ definition: selectedDefinition, metrics: adHocMetrics ?? undefined });
  }

  function handleStartAdHoc(metrics: MetricSelection[]) {
    closeBuilderModal();
    runCustomAnalysis({ metrics });
  }

  function handleSelectSuggestion(nextSymbol: string) {
    setSymbol(nextSymbol.toUpperCase());
    setIsSuggestionsOpen(false);
  }

  function handleSymbolBlur() {
    // Delay closing so a click on a suggestion still registers before the
    // dropdown unmounts.
    blurTimeoutRef.current = window.setTimeout(() => setIsSuggestionsOpen(false), 120);
  }

  function handleSymbolFocus() {
    if (blurTimeoutRef.current) {
      window.clearTimeout(blurTimeoutRef.current);
    }
    setIsSuggestionsOpen(true);
  }

  const canStartCustom = Boolean(selectedDefinition || adHocMetrics);
  const builderModalTitle =
    builderModalMode === "adhoc"
      ? "Einmalige Analyse"
      : builderModalMode === "edit"
        ? "Analyse bearbeiten"
        : "Neue Analyse erstellen";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "24px",
        padding: "10px 6px 20px",
      }}
    >
      <AnalyzeStickyBar
        symbol={normalizedSymbol}
        progress={analysisTab === "standard" ? progress : customProgress}
        result={analysisTab === "standard" ? result : null}
      />

      {pageErrorMessage ? (
        <div
          style={{
            padding: "14px 16px",
            borderRadius: "14px",
            background: theme.colors.dangerSoft,
            border: `1px solid ${theme.colors.dangerBorder}`,
            color: theme.colors.dangerText,
            fontSize: "0.98rem",
            lineHeight: 1.7,
          }}
        >
          {pageErrorMessage}
        </div>
      ) : null}

      {pageSuccessMessage ? (
        <div
          style={{
            padding: "14px 16px",
            borderRadius: "14px",
            background: theme.colors.successSoft,
            border: `1px solid ${theme.colors.successBorder}`,
            color: theme.colors.successText,
            fontSize: "0.98rem",
            lineHeight: 1.7,
          }}
        >
          {pageSuccessMessage}
        </div>
      ) : null}

      <section style={heroSection}>
        <div style={heroBadge}>Analyse</div>

        <h1
          style={{
            margin: "0 0 14px 0",
            // clamp() statt fix 3rem: skaliert auf sehr schmalen Screens
            // herunter statt die Zeile umzubrechen (RESPONSIVE.md R-P2-2).
            fontSize: "clamp(2rem, 6vw, 3rem)",
            lineHeight: 1.05,
            letterSpacing: "-0.045em",
            color: theme.colors.textPrimary,
          }}
        >
          Analysiere Aktien gezielt nach deinem Ansatz
        </h1>

        <p
          style={{
            margin: 0,
            maxWidth: "860px",
            color: theme.colors.textPrimary,
            fontSize: "1.12rem",
            lineHeight: 1.9,
          }}
        >
          Wähle ein Symbol und entscheide, ob du eine vorgefertigte Standard-Analyse
          oder deine eigene Kennzahlen-Kombination starten möchtest. Fortschritt und
          Ergebnis werden direkt auf dieser Seite geladen.
        </p>
      </section>

      {isFirstRun && !symbol ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: "16px",
            padding: "14px 16px",
            borderRadius: "14px",
            background: theme.colors.chromeSoft,
            border: `1px solid ${theme.colors.chromeBorder}`,
            color: theme.colors.textPrimary,
            fontSize: "0.95rem",
            lineHeight: 1.6,
          }}
        >
          <span>
            Neu hier? <strong>Starte mit AAPL</strong> — einer unserer Beispiel-Ticker — um
            zu sehen, wie eine Analyse mit nachvollziehbaren Quellenangaben aussieht.
          </span>
          <div style={{ display: "flex", alignItems: "center", gap: "10px", flexShrink: 0 }}>
            <Button type="button" variant="secondary" onClick={() => setSymbol("AAPL")}>
              Mit AAPL starten
            </Button>
            <button
              type="button"
              onClick={dismissOnboarding}
              aria-label="Hinweis schließen"
              style={{
                background: "none",
                border: "none",
                color: theme.colors.textMuted,
                cursor: "pointer",
                padding: "4px",
                display: "flex",
              }}
            >
              <X size={16} />
            </button>
          </div>
        </div>
      ) : null}

      <AnalyzeWorkspace
        symbol={symbol}
        onSymbolChange={setSymbol}
        onSymbolFocus={handleSymbolFocus}
        onSymbolBlur={handleSymbolBlur}
        isSuggestionsOpen={isSuggestionsOpen}
        isLoadingSymbols={isLoadingSymbols}
        filteredSuggestions={filteredSuggestions}
        onSelectSuggestion={handleSelectSuggestion}
        isFavorited={isFavorited}
        onToggleFavorite={handleToggleFavorite}
        analysisTab={analysisTab}
        onSelectAnalysisTab={setAnalysisTab}
        modeOptions={modeOptions}
        selectedMode={selectedMode}
        onSelectMode={setSelectedMode}
        customDefinitions={definitions}
        selectedDefinitionId={selectedDefinition?.id ?? null}
        onSelectCustomDefinition={handleSelectCustomDefinition}
        onEditCustomDefinition={handleEditDefinition}
        onDeleteCustomDefinition={handleDeleteDefinition}
        canCreateNewDefinition={canCreateNew}
        onCreateNewDefinition={handleCreateNewDefinition}
        onRunAdHoc={handleRunAdHoc}
        selectedFrequency={selectedFrequency}
        onFrequencyChange={setSelectedFrequency}
        startButton={
          <Button
            type="button"
            variant="cta"
            onClick={() => (analysisTab === "standard" ? handleStartAnalysis() : handleStartCustomAnalysis())}
            disabled={
              isStartingAnalysis ||
              !normalizedSymbol ||
              (analysisTab === "individuell" && !canStartCustom)
            }
            style={{ padding: "14px 24px", fontSize: "1rem" }}
          >
            {isStartingAnalysis
              ? "Analyse wird gestartet..."
              : isAnalysisLimitReached
              ? "Kontingent aufgebraucht — mehr erfahren"
              : "Analyse starten"}
          </Button>
        }
      />

      <section ref={resultsSectionRef} style={resultsSection}>
        <div style={sectionEyebrow}>Ergebnis</div>
        <div style={{ display: "flex", alignItems: "baseline", gap: "16px", flexWrap: "wrap" }}>
          <h2 style={resultsTitle}>Analyseausgabe</h2>
          {normalizedSymbol ? <span style={resultsSymbolBadge}>{normalizedSymbol}</span> : null}
          {analysisTab === "individuell" && customResult ? (
            <LivePriceBadge symbol={customResult.symbol} size="md" />
          ) : null}
          {analysisTab === "standard" && result ? (
            <SourceBadge symbol={result.symbol} frequency={selectedFrequency} />
          ) : null}
          {analysisTab === "individuell" && customResult ? (
            <SourceBadge symbol={customResult.symbol} />
          ) : null}
        </div>

        {isFirstRun &&
        ((analysisTab === "standard" && result) || (analysisTab === "individuell" && customResult)) ? (
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "16px",
              padding: "12px 16px",
              borderRadius: "14px",
              background: theme.colors.chromeSoft,
              border: `1px solid ${theme.colors.chromeBorder}`,
              color: theme.colors.textPrimary,
              fontSize: "0.92rem",
              lineHeight: 1.6,
            }}
          >
            <span>
              So siehst du, woher jede Zahl kommt: das Badge oben zeigt Quelle und Stand
              der Daten zu diesem Ergebnis.
            </span>
            <button
              type="button"
              onClick={dismissOnboarding}
              aria-label="Hinweis schließen"
              style={{
                background: "none",
                border: "none",
                color: theme.colors.textMuted,
                cursor: "pointer",
                padding: "4px",
                display: "flex",
                flexShrink: 0,
              }}
            >
              <X size={16} />
            </button>
          </div>
        ) : null}

        <div style={resultBox}>
          {analysisTab === "standard" ? (
            result ? (
              <AnalyzeResultsDashboard data={result} />
            ) : progress ? (
              <div style={resultPlaceholder}>
                Analyse läuft — den Fortschritt siehst du oben in der Leiste.
                Sobald sie abgeschlossen ist, erscheint die Auswertung hier.
              </div>
            ) : (
              <div style={resultPlaceholder}>Noch keine Analyse gestartet.</div>
            )
          ) : customResult ? (
            <CustomAnalysisResultsList catalog={catalog} result={customResult} />
          ) : customProgress ? (
            <div style={resultPlaceholder}>
              Analyse läuft — den Fortschritt siehst du oben in der Leiste.
              Sobald sie abgeschlossen ist, erscheint die Auswertung hier.
            </div>
          ) : (
            <div style={resultPlaceholder}>
              Wähle eine eigene Analyse, oder erstelle über "Einmalige Analyse" eine
              neue Kennzahlen-Kombination.
            </div>
          )}
        </div>
      </section>

      <Modal isOpen={builderModalMode !== null} onClose={closeBuilderModal} title={builderModalTitle} maxWidth="900px">
        {builderModalMode === "adhoc" ? (
          <AdHocAnalysisPanel
            catalog={catalog}
            isLoadingCatalog={isLoadingCatalog}
            symbol={symbol}
            onSymbolChange={setSymbol}
            onSymbolFocus={handleSymbolFocus}
            onSymbolBlur={handleSymbolBlur}
            isSuggestionsOpen={isSuggestionsOpen}
            isLoadingSymbols={isLoadingSymbols}
            filteredSuggestions={filteredSuggestions}
            onSelectSuggestion={handleSelectSuggestion}
            isFavorited={isFavorited}
            onToggleFavorite={handleToggleFavorite}
            isStarting={isStartingAnalysis}
            isLimitReached={isAnalysisLimitReached}
            onStart={handleStartAdHoc}
          />
        ) : (
          <DefinitionBuilder
            catalog={catalog}
            isLoadingCatalog={isLoadingCatalog}
            initialName={builderModalMode === "edit" ? editingDefinition?.name : undefined}
            initialMetrics={builderModalMode === "edit" ? editingDefinition?.metrics : undefined}
            isSaving={isSavingDefinition}
            onSave={handleBuilderSave}
            onCancel={closeBuilderModal}
          />
        )}
      </Modal>

      <QuotaExceededModal
        isOpen={isQuotaModalOpen}
        onClose={() => setIsQuotaModalOpen(false)}
        resetDate={quotaModalResetDate}
      />
    </div>
  );
}

const heroSection = {
  background: theme.glass.elevated.background,
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  borderRadius: theme.radius.lg,
  padding: "34px 34px 36px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
};

const heroBadge = {
  display: "inline-block",
  marginBottom: "16px",
  padding: "8px 12px",
  borderRadius: theme.radius.pill,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.chrome,
  fontSize: "0.86rem",
  fontWeight: 700,
  letterSpacing: "0.03em",
};

const sectionEyebrow = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const resultsSection = {
  background: theme.colors.panelAlt,
  borderRadius: theme.radius.lg,
  padding: "34px",
  border: `1px solid ${theme.colors.border}`,
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.35)",
};

const resultsTitle = {
  margin: "10px 0 24px 0",
  fontSize: "2rem",
  lineHeight: 1.15,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.03em",
};

const resultsSymbolBadge = {
  padding: "6px 12px",
  borderRadius: theme.radius.pill,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
  fontWeight: 800,
  fontSize: "1.1rem",
  letterSpacing: "0.02em",
};

const resultBox = {
  padding: "18px 20px",
  borderRadius: theme.radius.lg,
  background: theme.colors.panel,
  border: `1px solid ${theme.colors.borderSubtle}`,
  overflowX: "auto" as const,
};

const resultPlaceholder = {
  color: theme.colors.textSecondary,
  fontSize: "1rem",
  lineHeight: 1.8,
};

export default AnalyzePage;
