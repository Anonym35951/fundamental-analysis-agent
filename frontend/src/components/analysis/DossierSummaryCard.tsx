import ParallaxCard from "../ui/ParallaxCard";
import { theme } from "../ui/theme";
import LivePriceBadge from "../shared/LivePriceBadge";

type Props = {
  symbol: string;
  bestAssessment: string;
  total: number;
  positive: number;
  negative: number;
  neutral: number;
};

/** Compact "single-glance" header for the dossier — symbol + the most
 * load-bearing fact (best assessment) resolve in the title row, the stat
 * row stays a slim secondary line rather than competing for attention. */
export default function DossierSummaryCard({ symbol, bestAssessment, total, positive, negative, neutral }: Props) {
  return (
    <ParallaxCard style={summaryCard}>
      <div style={titleRow}>
        <div>
          <div style={eyebrow}>Executive Summary</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: "14px", flexWrap: "wrap" }}>
            <h3 style={summaryTitle}>{symbol.toUpperCase()}</h3>
            <LivePriceBadge symbol={symbol} size="md" />
          </div>
        </div>

        {bestAssessment ? (
          <div style={assessmentPill}>{bestAssessment}</div>
        ) : null}
      </div>

      <div style={summaryStatsRow}>
        <SummaryStat label="Analysen" value={total} />
        <SummaryStat label="Erfüllt" value={positive} tone="good" />
        <SummaryStat label="Kritisch" value={negative} tone="bad" />
        <SummaryStat label="Neutral" value={neutral} />
      </div>
    </ParallaxCard>
  );
}

function SummaryStat({ label, value, tone }: { label: string; value: number; tone?: "good" | "bad" }) {
  return (
    <div style={summaryStat}>
      <span style={summaryStatLabel}>{label}</span>
      <strong
        style={{
          ...summaryStatValue,
          color: tone === "good" ? theme.colors.successText : tone === "bad" ? theme.colors.dangerText : theme.colors.textPrimary,
        }}
      >
        {value}
      </strong>
    </div>
  );
}

const summaryCard = {
  display: "flex",
  flexDirection: "column" as const,
  gap: theme.spacing(4),
};

const titleRow = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  flexWrap: "wrap" as const,
  gap: "12px",
};

const eyebrow = {
  color: theme.colors.chrome,
  fontSize: "0.82rem",
  fontWeight: 800,
  letterSpacing: "0.04em",
  textTransform: "uppercase" as const,
};

const summaryTitle = {
  margin: "8px 0 0 0",
  color: theme.colors.textPrimary,
  fontSize: "2rem",
  lineHeight: 1.1,
  letterSpacing: "-0.04em",
};

const assessmentPill = {
  padding: "10px 16px",
  borderRadius: theme.radius.pill,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
  fontSize: "0.92rem",
  fontWeight: 700,
  maxWidth: "420px",
};

const summaryStatsRow = {
  display: "flex",
  flexWrap: "wrap" as const,
  gap: "10px",
};

const summaryStat = {
  flex: "1 1 110px",
  padding: "12px 14px",
  borderRadius: theme.radius.md,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.borderSubtle}`,
  display: "flex",
  flexDirection: "column" as const,
  gap: "4px",
};

const summaryStatLabel = {
  color: theme.colors.textMuted,
  fontSize: "0.74rem",
  fontWeight: 800,
};

const summaryStatValue = {
  fontSize: "1.25rem",
  fontWeight: 900,
};
