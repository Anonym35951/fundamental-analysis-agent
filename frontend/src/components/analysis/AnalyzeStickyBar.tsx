import type { FullResult, Progress } from "../../types/analysis";
import { theme } from "../ui/theme";
import ProgressReadout from "./ProgressReadout";

type Props = {
  symbol: string;
  progress: Progress | null;
  result: FullResult | null;
};

export default function AnalyzeStickyBar({ symbol, progress, result }: Props) {
  const percent =
    progress && progress.total > 0
      ? Math.round((progress.done / progress.total) * 100)
      : 0;

  const show = !!result || !!progress;

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
          padding: "12px 14px",
          borderRadius: theme.radius.lg,
          border: `1px solid ${theme.glass.border}`,
          background: theme.glass.background,
          backdropFilter: `blur(${theme.glass.blur})`,
          WebkitBackdropFilter: `blur(${theme.glass.blur})`,
          boxShadow: theme.glass.shadow,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <div
            style={{
              fontWeight: 800,
              letterSpacing: "-0.02em",
              color: theme.colors.textPrimary,
              fontSize: "0.98rem",
            }}
          >
            ComAnalysis
          </div>

          <div
            style={{
              fontSize: 12,
              color: theme.colors.textSecondary,
              display: "flex",
              alignItems: "center",
              gap: 6,
              flexWrap: "wrap",
            }}
          >
            {symbol ? (
              <span
                style={{
                  padding: "5px 9px",
                  borderRadius: theme.radius.pill,
                  background: theme.colors.chromeSoft,
                  border: `1px solid ${theme.colors.chromeBorder}`,
                  color: theme.colors.textPrimary,
                  fontWeight: 700,
                }}
              >
                {symbol.toUpperCase()}
              </span>
            ) : null}

            {progress ? (
              <span style={{ color: theme.colors.textMuted }}>
                • {progress.status === "running" ? "läuft" : progress.status} •{" "}
                {percent}%
              </span>
            ) : null}
          </div>
        </div>

        {progress && progress.status === "running" ? (
          <ProgressReadout percent={percent} status={progress.status} currentStep={progress.current} />
        ) : null}
      </div>
    </div>
  );
}
