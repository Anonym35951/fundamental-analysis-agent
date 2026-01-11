// richtig: zeigt auf Backend (Web Service)
const API_BASE = "https://fundamental-analysis-agent.onrender.com";

// =====================================================
// FULL ANALYSIS (Job + Progress) – UNVERÄNDERT
// =====================================================
export async function startFullAnalysis(symbol: string) {
  const res = await fetch(
    `${API_BASE}/full/start?symbol=${encodeURIComponent(symbol)}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error("Failed to start analysis");
  return res.json(); // { job_id, symbol, total }
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

type SingleCallResult = {
  results: Record<string, any>;
  total: number;
};

// =====================================================
// HELPERS
// =====================================================
async function getJson(url: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Request failed: ${res.status}`);
  return res.json();
}

// =====================================================
// 🔁 SYNC SINGLE ANALYSES (BESTEHEND)
// =====================================================
async function fetchWithFrequencies(
  symbol: string,
  basePath: string,
  analysisName: string,
  supportsQuarterly: boolean
): Promise<SingleCallResult> {
  const results: Record<string, any> = {};

  const annual = await getJson(
    `${API_BASE}${basePath}?symbol=${encodeURIComponent(symbol)}&frequency=annual`
  );
  results[`${analysisName}|annual`] = annual;

  if (supportsQuarterly) {
    const quarterly = await getJson(
      `${API_BASE}${basePath}?symbol=${encodeURIComponent(symbol)}&frequency=quarterly`
    );
    results[`${analysisName}|quarterly`] = quarterly;
  }

  return { results, total: Object.keys(results).length };
}

export async function runSingleAnalysis(
  symbol: string,
  mode: Exclude<AnalysisMode, "full">
): Promise<SingleCallResult> {
  const s = symbol.trim().toUpperCase();

  switch (mode) {
    case "wachstumswerte":
      return fetchWithFrequencies(s, "/analyze/wachstumswerte", "Wachstumswerte", true);

    case "typische-zykliker":
      return fetchWithFrequencies(s, "/analyze/typische-zykliker", "Typische Zykliker", true);

    case "turnarounds":
      return fetchWithFrequencies(s, "/analyze/turnarounds", "Zyklische Turnarounds", true);

    case "optionality": {
      const annual = await getJson(
        `${API_BASE}/analyze/optionality?symbol=${encodeURIComponent(s)}&frequency=annual`
      );
      return { results: { "Optionality|annual": annual }, total: 1 };
    }

    case "dividendenwerte": {
      const annual = await getJson(
        `${API_BASE}/analyze/dividendenwerte?symbol=${encodeURIComponent(s)}`
      );
      return { results: { "Dividendenwerte|annual": annual }, total: 1 };
    }

    case "average-grower": {
      const annual = await getJson(
        `${API_BASE}/analyze/average-grower?symbol=${encodeURIComponent(s)}`
      );
      return { results: { "Average Grower|annual": annual }, total: 1 };
    }

    case "asset-play": {
      const annual = await getJson(
        `${API_BASE}/analyze/asset-play?symbol=${encodeURIComponent(s)}&frequency=annual`
      );
      return { results: { "Asset Play|annual": annual }, total: 1 };
    }

    default:
      throw new Error("Unknown analysis mode");
  }
}

// =====================================================
// 🆕 JOB-BASED SINGLE ANALYSES (MIT PROGRESS)
// =====================================================
export async function startSingleAnalysisJob(
  symbol: string,
  mode: Exclude<AnalysisMode, "full">,
  frequency: "annual" | "quarterly" = "annual"
) {
  const res = await fetch(
    `${API_BASE}/analyze/${mode}/start?symbol=${encodeURIComponent(symbol)}&frequency=${frequency}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error("Failed to start single analysis");
  return res.json(); // { job_id, symbol, mode }
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