import { apiRequest } from "./client";
import type { Progress, FullResult } from "../types/analysis";

type JobStartResponse = {
  job_id?: string;
  jobId?: string;
  id?: string;
  [key: string]: unknown;
};

// =====================================================
// FULL ANALYSIS (Job + Progress)
// =====================================================
export async function startFullAnalysis(symbol: string) {
  return apiRequest<JobStartResponse>(
    `/full/start?symbol=${encodeURIComponent(symbol)}`,
    { method: "POST" }
  );
}

export async function getProgress(jobId: string) {
  return apiRequest<Progress>(`/full/full/${jobId}/progress`);
}

export async function getResult(jobId: string) {
  return apiRequest<FullResult>(`/full/full/${jobId}/result`);
}

// =====================================================
// ANALYSIS MODES
// =====================================================
export type AnalysisMode =
  | "full"
  | "wachstumswerte"
  | "dividendenwerte"
  | "average-grower"
  | "typische-zykliker"
  | "turnarounds"
  | "optionality"
  | "asset-play";

// =====================================================
// SYMBOL SEARCH (Autocomplete / Validation)
// =====================================================
export type SymbolMeta = {
  symbol: string;
  name: string;
  sectors: string[];
};

/** Durchsucht das volle NYSE+NASDAQ-Symbol-Universum serverseitig (siehe
 * api/routes/analyze.py: search_symbols) - ein leerer Query liefert eine
 * kuratierte Popular-Liste statt aller ~6-7k Zeilen. */
export async function searchSymbols(query: string, limit = 20): Promise<SymbolMeta[]> {
  const params = new URLSearchParams();
  if (query) params.set("query", query);
  params.set("limit", String(limit));
  return apiRequest<SymbolMeta[]>(`/analyze/symbols?${params.toString()}`, { skipAuth: true });
}

// =====================================================
// LOCAL FALLBACK SYMBOL LIST
// =====================================================
import { LOCAL_SYMBOLS } from "../data/symbols";

function filterLocalSymbols(query: string, limit: number): SymbolMeta[] {
  const normalized = query.trim().toUpperCase();
  if (!normalized) return LOCAL_SYMBOLS.slice(0, limit);
  return LOCAL_SYMBOLS.filter(
    (entry) =>
      entry.symbol.toUpperCase().includes(normalized) ||
      entry.name.toUpperCase().includes(normalized)
  ).slice(0, limit);
}

export type SymbolSearchResult = {
  entries: SymbolMeta[];
  /** true, wenn die Backend-Suche fehlgeschlagen ist (Netzwerkfehler,
   * Rate-Limit, 5xx) und stattdessen auf die statische 23-Symbol-Liste
   * zurückgefallen wurde (EVOLVING.md EV-013) - Aufrufer sollen das nicht
   * mehr still verschlucken, sondern dem Nutzer sichtbar machen. */
  degraded: boolean;
};

export async function searchSymbolsSafe(query: string, limit = 20): Promise<SymbolSearchResult> {
  try {
    return { entries: await searchSymbols(query, limit), degraded: false };
  } catch {
    console.warn("Backend symbol search unavailable – using local fallback");
    return { entries: filterLocalSymbols(query, limit), degraded: true };
  }
}

// =====================================================
// JOB-BASED SINGLE ANALYSES
// =====================================================
export async function startSingleAnalysisJob(
  symbol: string,
  mode: Exclude<AnalysisMode, "full">,
  frequency: "annual" | "quarterly" = "annual"
) {
  return apiRequest<JobStartResponse>(
    `/analyze/${mode}/start?symbol=${encodeURIComponent(symbol)}&frequency=${frequency}`,
    { method: "POST" }
  );
}

export async function getSingleProgress(
  mode: Exclude<AnalysisMode, "full">,
  jobId: string
) {
  return apiRequest<Progress>(`/analyze/${mode}/${jobId}/progress`);
}

export async function getSingleResult(
  mode: Exclude<AnalysisMode, "full">,
  jobId: string
) {
  return apiRequest<FullResult>(`/analyze/${mode}/${jobId}/result`);
}

// =====================================================
// ANALYSIS HISTORY
// =====================================================
export type AnalysisHistoryEntry = {
  id: number;
  job_id: string;
  symbol: string;
  mode: string;
  frequency: string | null;
  status: "running" | "done" | "error";
  definition_id: number | null;
  created_at: string;
};

export type AnalysisHistorySnapshot = AnalysisHistoryEntry & {
  result_snapshot: Record<string, unknown> | null;
};

export async function getAnalysisHistory(): Promise<AnalysisHistoryEntry[]> {
  return apiRequest<AnalysisHistoryEntry[]>("/analyze/history");
}

export async function getAnalysisHistorySnapshot(
  historyId: number
): Promise<AnalysisHistorySnapshot> {
  return apiRequest<AnalysisHistorySnapshot>(`/analyze/history/${historyId}/snapshot`);
}
