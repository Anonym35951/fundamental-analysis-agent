// frontend/src/pages/LandingPage.tsx
import { useEffect, useMemo, useRef, useState } from "react";
import {
  startFullAnalysis,
  getProgress,
  getResult,
  startSingleAnalysisJob,
  getSingleProgress,
  getSingleResult,
  type AnalysisMode,
} from "../api/client";
import ResultsView from "../components/ResultsView";
import Background from "../components/Background";
import StickyBar from "../components/StickyBar";
import type { FullResult, Progress as ApiProgress } from "../types/api";

type Progress = ApiProgress;

type Health = "all" | "good" | "bad" | "neutral";
type Freq = "all" | "annual" | "quarterly";

function splitKey(key: string) {
  const [name, freq] = key.split("|");
  return { name: name ?? key, freq: (freq ?? "").toLowerCase() as "annual" | "quarterly" | "" };
}

// 🔍 rekursive Suche (Kennzahlen / Payload)
function payloadContainsQuery(payload: any, query: string): boolean {
  if (!payload || !query) return false;
  const q = query.toLowerCase();

  const walk = (obj: any): boolean => {
    if (obj == null) return false;
    if (["string", "number", "boolean"].includes(typeof obj)) {
      return String(obj).toLowerCase().includes(q);
    }
    if (Array.isArray(obj)) return obj.some(walk);
    if (typeof obj === "object") {
      return Object.entries(obj).some(([k, v]) => k.toLowerCase().includes(q) || walk(v));
    }
    return false;
  };

  return walk(payload);
}

function calcAnalysisHealth(payload: any): Exclude<Health, "all"> {
  if (!payload || typeof payload !== "object") return "neutral";

  const keys = Object.keys(payload).filter(
    (k) => !["symbol", "frequency", "overall_assessment", "message", "crv"].includes(k)
  );

  let hasTrue = false;
  for (const k of keys) {
    const v = payload[k];
    if (v?.meets_criterion === false) return "bad";
    if (v?.meets_criterion === true) hasTrue = true;
  }

  return hasTrue ? "good" : "neutral";
}

export default function LandingPage() {
  const [symbol, setSymbol] = useState("");
  const [analysisMode, setAnalysisMode] = useState<AnalysisMode>("full");
  const [jobId, setJobId] = useState<string | null>(null);

  const [progress, setProgress] = useState<Progress | null>(null);
  const [result, setResult] = useState<FullResult | null>(null);

  const [loading, setLoading] = useState(false);

  const [query, setQuery] = useState("");
  const [freq, setFreq] = useState<Freq>("all");
  const [health, setHealth] = useState<Health>("all");

  const intervalRef = useRef<number | null>(null);

  // ✅ Der Mode des *aktuell gestarteten Jobs* (nicht Dropdown-Status!)
  const activeJobModeRef = useRef<AnalysisMode>("full");

  async function onStart() {
    setResult(null);
    setProgress(null);
    setJobId(null);

    const clean = symbol.trim().toUpperCase();
    if (!clean) return;

    setLoading(true);

    try {
      // ✅ aktiven Job-Mode festnageln
      activeJobModeRef.current = analysisMode;

      if (analysisMode !== "full") {
        const data = await startSingleAnalysisJob(clean, analysisMode);
        setJobId(data.job_id);
        return;
      }

      const data = await startFullAnalysis(clean);
      setJobId(data.job_id);
    } catch (e: any) {
      console.error(e);
      setLoading(false);
    }
  }

  // 🔁 Polling: NUR abhängig vom jobId (nicht vom Dropdown!)
  useEffect(() => {
    if (!jobId) return;

    // vorheriges Intervall sauber stoppen
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    const poll = async () => {
      try {
        const activeMode = activeJobModeRef.current;

        // 🆕 SINGLE
        if (activeMode !== "full") {
          const p = await getSingleProgress(activeMode, jobId);
          setProgress(p);

          if (p.status === "done") {
            if (intervalRef.current !== null) {
              window.clearInterval(intervalRef.current);
              intervalRef.current = null;
            }

            const r = await getSingleResult(activeMode, jobId);
            setResult(r);
            setLoading(false);
          } else if (p.status === "error") {
            if (intervalRef.current !== null) {
              window.clearInterval(intervalRef.current);
              intervalRef.current = null;
            }
            setLoading(false);
          }
          return;
        }

        // 🟢 FULL
        const p = await getProgress(jobId);
        setProgress(p);

        if (p.status === "done") {
          if (intervalRef.current !== null) {
            window.clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          setResult(await getResult(jobId));
          setLoading(false);
        } else if (p.status === "error") {
          if (intervalRef.current !== null) {
            window.clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
          setLoading(false);
        }
      } catch (e: any) {
        // ✅ 404 beim Initial-Poll / Race Conditions ignorieren
        const msg = String(e?.message ?? "");
        if (msg.includes("404")) return;

        console.error(e);
        setLoading(false);
      }
    };

    poll();
    intervalRef.current = window.setInterval(poll, 700);

    // ✅ Cleanup muss void zurückgeben
    return () => {
      if (intervalRef.current !== null) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [jobId]);

  // 🔎 Filter
  const filteredEntries = useMemo(() => {
    if (!result) return [];
    const q = query.toLowerCase();

    return Object.entries(result.results ?? {}).filter(([key, payload]) => {
      const { name, freq: f } = splitKey(key);

      if (q) {
        const meta = `${name} ${f} ${payload?.overall_assessment ?? ""} ${payload?.message ?? ""}`.toLowerCase();
        if (!meta.includes(q) && !payloadContainsQuery(payload, q)) return false;
      }

      if (freq !== "all" && f !== freq) return false;
      if (health !== "all" && calcAnalysisHealth(payload) !== health) return false;
      return true;
    });
  }, [result, query, freq, health]);

  const [leftResult, rightResult] = useMemo(() => {
    if (!result) return [null, null] as const;
    const mid = Math.ceil(filteredEntries.length / 2);

    const make = (subset: [string, any][]) =>
      ({ ...result, results: Object.fromEntries(subset), total: subset.length } as FullResult);

    return [make(filteredEntries.slice(0, mid)), make(filteredEntries.slice(mid))] as const;
  }, [result, filteredEntries]);

  return (
    <div style={{ minHeight: "100vh", padding: "56px 24px" }}>
      <Background />

      <StickyBar
        symbol={symbol}
        progress={progress}
        result={result}
        query={query}
        onQueryChange={setQuery}
        freq={freq}
        onFreqChange={setFreq}
        health={health}
        onHealthChange={setHealth}
        onClearFilters={() => {
          setQuery("");
          setFreq("all");
          setHealth("all");
        }}
      />

      <div style={{ maxWidth: 1200, margin: "0 auto" }}>
        {/* Titel + Controls */}
        <div style={{ maxWidth: 520, margin: "0 auto", textAlign: "center", marginBottom: 18 }}>
          <h1
            style={{
              fontSize: 36,
              fontWeight: 700,
              letterSpacing: "-0.02em",
              marginBottom: 6,
              textShadow: "0 1px 20px rgba(255, 255, 255, 0.06)",
            }}
          >
            Clarity over Noise
          </h1>

          <div style={{ fontSize: 13, letterSpacing: "0.18em", textTransform: "uppercase", opacity: 0.75 }}>
            Analyze. Decide. Hold.
          </div>

          <div style={{ marginTop: 14 }}>
            <select
              value={analysisMode}
              onChange={(e) => setAnalysisMode(e.target.value as AnalysisMode)}
              style={{
                width: "100%",
                padding: "10px 12px",
                borderRadius: 10,
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.14)",
                color: "white",
              }}
            >
              <option value="full">Full Analysis</option>
              <option value="wachstumswerte">Wachstumswerte</option>
              <option value="average-grower">Average Grower</option>
              <option value="dividendenwerte">Dividendenwerte</option>
              <option value="typische-zykliker">Typische Zykliker</option>
              <option value="turnarounds">Turnarounds</option>
              <option value="optionality">Optionality</option>
              <option value="asset-play">Asset Play</option>
            </select>
          </div>

          <div style={{ display: "flex", gap: 10, marginTop: 12 }}>
            <input
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              placeholder="Symbol (z.B. BABA)"
              style={{
                flex: 1,
                padding: "12px 14px",
                borderRadius: 12,
                border: "1px solid rgba(255,255,255,0.14)",
                background: "rgba(255,255,255,0.06)",
                color: "white",
              }}
            />
            <button onClick={onStart} disabled={loading}>
              {loading ? "Läuft…" : "Start"}
            </button>
          </div>
        </div>

        {result && (
          <div className="twoColResults" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 18 }}>
            <div>{leftResult && <ResultsView data={leftResult} query={query} />}</div>
            <div>{rightResult && <ResultsView data={rightResult} query={query} />}</div>
          </div>
        )}
      </div>
    </div>
  );
}