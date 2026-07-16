import { createContext } from "react";
import type { CustomMetricResult, MetricSelection } from "../types/customAnalysis";
import type { Frequency } from "../types/frequency";

export type CompareFrequency = Frequency;

export type CompanyResult = {
  symbol: string;
  status: "running" | "done" | "error";
  error?: string | null;
  /** Only the keys that were actually covered by the last successful fetch
   * — compared against the current metric selection to decide staleness. */
  metrics: Record<string, CustomMetricResult>;
  /** Params each metric in `metrics` was last fetched with, so a param
   * change (e.g. switching the `multiple` of a Kursziel metric) is detected
   * as stale even though the key itself didn't change. */
  metricParams?: Record<string, Record<string, unknown>>;
  frequency?: CompareFrequency;
  /** ISO-Berichtswährung der Fundamentaldaten (EVOLVING.md EV-021/022) -
   * `null`/fehlend, wenn unbestimmbar oder noch nicht geladen. */
  reporting_currency?: string | null;
};

export type CompareContextValue = {
  metrics: MetricSelection[];
  companies: CompanyResult[];
  frequency: CompareFrequency;
  hasStarted: boolean;
  setMetrics: (metrics: MetricSelection[]) => void;
  setFrequency: (frequency: CompareFrequency) => void;
  addCompany: (symbol: string) => void;
  removeCompany: (symbol: string) => void;
  startComparison: () => void;
  clearAll: () => void;
};

// Eigene Datei ohne Komponenten-Export, damit weder useCompare.tsx
// (CompareProvider) noch useCompareContext.ts (Hook) selbst einen
// Nicht-Komponenten-Wert exportieren (react-refresh/only-export-components,
// LAUNCH_AUDIT.md P2-10).
export const CompareContext = createContext<CompareContextValue | null>(null);
