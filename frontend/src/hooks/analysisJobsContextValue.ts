import { createContext } from "react";
import type { AnalysisMode } from "../api/analysis";
import type { CustomAnalysisDefinition, CustomAnalysisResult, MetricSelection } from "../types/customAnalysis";
import type { FullResult, Progress } from "../types/analysis";

export type AnalysisJobKind = "single" | "full" | "custom";

export type AnalyzeJobRecord = {
  id: string; // = job_id
  kind: AnalysisJobKind;
  symbol: string;
  mode?: AnalysisMode;
  frequency?: "annual" | "quarterly";
  /** Für den Notification-Text, z.B. "Vollanalyse" oder der Name einer
   * gespeicherten eigenen Analyse. */
  modeLabel?: string;
  definitionId?: number | null;
  definitionName?: string | null;
  status: "running" | "done" | "error";
  progress: Progress | null;
  result: FullResult | CustomAnalysisResult | null;
  error: string | null;
  startedAt: number;
  finishedAt: number | null;
  /** Edge-Trigger-Guard: die Fertig-Notification feuert genau einmal pro
   * Fertigstellung, auch wenn der Job-Record danach noch weiterlebt. */
  notified: boolean;
};

export type AnalysisJobsContextValue = {
  jobs: AnalyzeJobRecord[];
  startFullOrSingleJob: (params: {
    symbol: string;
    mode: AnalysisMode;
    frequency: "annual" | "quarterly";
    modeLabel: string;
  }) => Promise<string>;
  startCustomJob: (params: {
    symbol: string;
    metrics?: MetricSelection[];
    definition?: CustomAnalysisDefinition | null;
  }) => Promise<string>;
  getJob: (id: string | null) => AnalyzeJobRecord | undefined;
};

// Eigene Datei ohne Komponenten-Export, damit weder useAnalysisJobs.tsx
// (AnalysisJobsProvider) noch useAnalysisJobsContext.ts (Hook) selbst einen
// Nicht-Komponenten-Wert exportieren (react-refresh/only-export-components,
// LAUNCH_AUDIT.md P2-10) — spiegelt compareContextValue.ts.
export const AnalysisJobsContext = createContext<AnalysisJobsContextValue | null>(null);
