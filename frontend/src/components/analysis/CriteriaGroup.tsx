import MetricResultCard from "../metrics/MetricResultCard";
import { theme } from "../ui/theme";
import type { Criterion } from "./analysisResultUtils";

type Props = {
  label: string;
  tone: "danger" | "success" | "neutral";
  criteria: Criterion[];
  query?: string;
};

const toneColor: Record<Props["tone"], string> = {
  danger: theme.colors.danger,
  success: theme.colors.success,
  neutral: theme.colors.textMuted,
};

export default function CriteriaGroup({ label, tone, criteria, query }: Props) {
  if (criteria.length === 0) return null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
      <div
        style={{
          fontSize: "0.74rem",
          fontWeight: 800,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: toneColor[tone],
        }}
      >
        {label} · {criteria.length}
      </div>

      {criteria.map((criterion) => (
        <MetricResultCard
          key={criterion.key}
          metricKey={criterion.key}
          value={criterion.value}
          meetsCriterion={criterion.meets}
          message={criterion.message}
          query={query}
          style={{ background: theme.colors.panelAlt }}
        />
      ))}
    </div>
  );
}
