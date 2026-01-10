//frontend/src/components/StickyBar.tsx
import type { FullResult, Progress } from "../types/api";

type Health = "all" | "good" | "bad" | "neutral";
type Freq = "all" | "annual" | "quarterly";

type Props = {
  symbol: string;
  progress: Progress | null;
  result: FullResult | null;

  query: string;
  onQueryChange: (v: string) => void;

  freq: Freq;
  onFreqChange: (v: Freq) => void;

  health: Health;
  onHealthChange: (v: Health) => void;

  onClearFilters: () => void;
};

export default function StickyBar({
  symbol,
  progress,
  result,
  query,
  onQueryChange,
  freq,
  onFreqChange,
  health,
  onHealthChange,
  onClearFilters,
}: Props) {
  const percent =
    progress && progress.total > 0
      ? Math.round((progress.done / progress.total) * 100)
      : 0;

  const show = !!result || !!progress || query.length > 0 || freq !== "all" || health !== "all";

  if (!show) return null;

  return (
    <div
      style={{
        position: "sticky",
        top: 16,
        zIndex: 50,
        marginBottom: 18,
      }}
    >
      <div
        style={{
          width: "100%",
          maxWidth: 1200,
          margin: "0 auto",
          padding: "10px 12px",
          borderRadius: 16,
          border: "1px solid rgba(255,255,255,0.14)",
          background: "rgba(255,255,255,0.06)",
          backdropFilter: "blur(12px)",
          WebkitBackdropFilter: "blur(12px)",
          boxShadow: "0 18px 50px rgba(0,0,0,0.35)",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
          flexWrap: "wrap",
        }}
      >
        {/* Left: Title + status */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ fontWeight: 700, letterSpacing: "-0.02em" }}>
            AIAgent
          </div>
          <div style={{ fontSize: 12, opacity: 0.8 }}>
            {symbol ? <span>• {symbol.toUpperCase()}</span> : null}
            {progress ? (
              <span>
                {" "}
                • {progress.status === "running" ? "läuft" : progress.status} •{" "}
                {percent}%
              </span>
            ) : null}
          </div>
        </div>

        {/* Middle: Search */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, flex: 1, minWidth: 260 }}>
          <input
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            placeholder="Suche (z.B. Zykliker, Optionality, Asset…)"
            style={{
              width: "100%",
              padding: "10px 12px",
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.14)",
              background: "rgba(0,0,0,0.18)",
              color: "white",
              outline: "none",
            }}
          />
        </div>

        {/* Right: Filters */}
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <select
            value={freq}
            onChange={(e) => onFreqChange(e.target.value as Freq)}
            style={selectStyle}
          >
            <option value="all">Alle Frequenzen</option>
            <option value="annual">Annual</option>
            <option value="quarterly">Quarterly</option>
          </select>

          <select
            value={health}
            onChange={(e) => onHealthChange(e.target.value as Health)}
            style={selectStyle}
          >
            <option value="all">Alle</option>
            <option value="good">Nur ✔ (positiv)</option>
            <option value="bad">Nur ✘ (kritisch)</option>
            <option value="neutral">Nur • (neutral)</option>
          </select>

          <button
            onClick={onClearFilters}
            style={{
              padding: "10px 12px",
              borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.14)",
              background: "rgba(255,255,255,0.10)",
              color: "white",
              cursor: "pointer",
            }}
            title="Filter zurücksetzen"
          >
            Reset
          </button>
        </div>

        {/* Progress mini-bar (optional) */}
        {progress && progress.status === "running" ? (
          <div style={{ width: "100%", marginTop: 6 }}>
            <div
              style={{
                height: 6,
                borderRadius: 999,
                background: "rgba(255,255,255,0.10)",
                overflow: "hidden",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${percent}%`,
                  background: "rgba(255,255,255,0.55)",
                  transition: "width 250ms ease",
                }}
              />
            </div>
            <div style={{ marginTop: 6, fontSize: 12, opacity: 0.8 }}>
              {progress.current ?? "…"}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  padding: "10px 10px",
  borderRadius: 12,
  border: "1px solid rgba(255,255,255,0.14)",
  background: "rgba(0,0,0,0.18)",
  color: "white",
  outline: "none",
};