// frontend/src/api/client.ts

// richtig: zeigt auf Backend (Web Service)
const API_BASE = "https://fundamental-analysis-agent.onrender.com";
// const API_BASE = "http://127.0.0.1:8000";

// =====================================================
// FULL ANALYSIS (Job + Progress)
// =====================================================
export async function startFullAnalysis(symbol: string) {
  const res = await fetch(
    `${API_BASE}/full/start?symbol=${encodeURIComponent(symbol)}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error("Failed to start analysis");
  return res.json();
}

export async function getProgress(jobId: string) {
  const res = await fetch(`${API_BASE}/full/full/${jobId}/progress`);
  if (!res.ok) throw new Error("Progress error");
  return res.json();
}

export async function getResult(jobId: string) {
  const res = await fetch(`${API_BASE}/full/full/${jobId}/result`);
  if (!res.ok) throw new Error("Result error");
  return res.json();
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
// 🆕 SYMBOL LIST (Autocomplete / Validation)
// =====================================================
export type SymbolMeta = {
  symbol: string;
  sectors: string[];
};

// 👉 Backend-Quelle (wie bisher)
export async function getSymbols(): Promise<SymbolMeta[]> {
  const res = await fetch(`${API_BASE}/analyze/symbols`);
  if (!res.ok) throw new Error("Failed to load symbols");
  return res.json();
}

// =====================================================
// 🆕 LOCAL FALLBACK SYMBOL LIST
// (wird z.B. beim App-Start oder bei Offline-Problemen genutzt)
// =====================================================
import { LOCAL_SYMBOLS } from "../data/symbols";

// Helper: bevorzugt Backend, fällt sauber zurück
export async function getSymbolsSafe(): Promise<SymbolMeta[]> {
  try {
    return await getSymbols();
  } catch {
    console.warn("⚠️ Backend symbols unavailable – using local fallback");
    return LOCAL_SYMBOLS;
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
  const res = await fetch(
    `${API_BASE}/analyze/${mode}/start?symbol=${encodeURIComponent(
      symbol
    )}&frequency=${frequency}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error("Failed to start single analysis");
  return res.json();
}

export async function getSingleProgress(
  mode: Exclude<AnalysisMode, "full">,
  jobId: string
) {
  const res = await fetch(`${API_BASE}/analyze/${mode}/${jobId}/progress`);
  if (!res.ok) throw new Error("Single progress error");
  return res.json();
}

export async function getSingleResult(
  mode: Exclude<AnalysisMode, "full">,
  jobId: string
) {
  const res = await fetch(`${API_BASE}/analyze/${mode}/${jobId}/result`);
  if (!res.ok) throw new Error("Single result error");
  return res.json();
}