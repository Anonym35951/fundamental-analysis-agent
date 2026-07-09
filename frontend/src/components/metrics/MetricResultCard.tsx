import type { CSSProperties } from "react";
import Badge from "../ui/Badge";
import InfoTooltip from "../ui/InfoTooltip";
import { theme } from "../ui/theme";
import { formatLabel, formatMetricValue, highlightText, isPassFail, type CriterionLike } from "./metricFormatting";

export type MetricResultCardProps = {
  /** Metric key, e.g. "calculate_roe" or a criterion key from the legacy
   * AgentAction payload shape. Used to derive a label and for formatting. */
  metricKey: string;
  /** Raw computed value (scalar, string, or any JSON-serializable shape). */
  value: unknown;
  /** Optional threshold criterion for a pass/fail badge. */
  criterion?: CriterionLike | null;
  /** Optional explicit pass/fail override (e.g. from AgentAction's
   * meets_criterion), takes precedence over criterion-based evaluation. */
  meetsCriterion?: boolean | null;
  error?: string | null;
  message?: string | null;
  query?: string;
  style?: CSSProperties;
};

/** Canonical "metric key -> value (+ optional series, + optional pass/fail
 * badge)" card. Shared between the redesigned Analyze results dashboard and
 * the new Custom Analysis runner so both render computed metrics identically. */
export default function MetricResultCard({
  metricKey,
  value,
  criterion,
  meetsCriterion,
  error,
  message,
  query,
  style,
}: MetricResultCardProps) {
  const resolvedMeets = meetsCriterion !== undefined ? meetsCriterion : isPassFail(criterion, value);

  return (
    <div
      style={{
        padding: "16px 18px",
        borderRadius: theme.radius.md,
        background: theme.colors.panel,
        border: `1px solid ${theme.colors.borderSubtle}`,
        display: "flex",
        flexDirection: "column",
        gap: "10px",
        ...style,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "12px" }}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "6px", color: theme.colors.textPrimary, fontSize: "0.92rem", fontWeight: 700 }}>
          {highlightText(formatLabel(metricKey), query)}
          <InfoTooltip metricKey={metricKey} />
        </span>

        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          {resolvedMeets !== null ? (
            <Badge tone={resolvedMeets ? "success" : "danger"}>{resolvedMeets ? "Erfüllt" : "Kritisch"}</Badge>
          ) : null}
          <span style={{ color: theme.colors.textSecondary, fontSize: "0.88rem", fontWeight: 700, whiteSpace: "nowrap" }}>
            {error ? "—" : formatMetricValue(value, metricKey)}
          </span>
        </div>
      </div>

      {message ? (
        <p style={{ margin: 0, color: theme.colors.textSecondary, fontSize: "0.86rem", lineHeight: 1.6 }}>
          {highlightText(message, query)}
        </p>
      ) : null}

      {error ? (
        <p style={{ margin: 0, color: theme.colors.dangerText, fontSize: "0.86rem", lineHeight: 1.6 }}>{highlightText(error, query)}</p>
      ) : null}
    </div>
  );
}
