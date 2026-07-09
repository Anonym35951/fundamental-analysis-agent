import { AlertTriangle } from "lucide-react";
import Badge from "../ui/Badge";
import { theme } from "../ui/theme";
import CrvTargetPanel from "../metrics/CrvTargetPanel";
import { highlightText } from "../metrics/metricFormatting";
import { extractCriteria, calculateScore, groupCriteriaByStatus, type AnalysisItem } from "./analysisResultUtils";
import CriteriaGroup from "./CriteriaGroup";

type Props = {
  item: AnalysisItem;
  query?: string;
};

export default function AnalysisFrequencyPanel({ item, query }: Props) {
  const criteria = extractCriteria(item.payload);
  const score = calculateScore(criteria);
  const grouped = groupCriteriaByStatus(criteria);

  return (
    <div style={frequencyPanel}>
      <div style={frequencyPanelHeader}>
        <div>
          <div style={frequencyLabel}>{item.frequency.toUpperCase()}</div>
          <div style={frequencyTitle}>{item.name}</div>
        </div>

        <Badge tone="accent">
          {score.positive}/{score.total} erfüllt
        </Badge>
      </div>

      {item.payload?.message ? <div style={analysisMessageBox}>{highlightText(String(item.payload.message), query)}</div> : null}
      {item.payload?.error ? (
        <div style={analysisErrorBox}>
          <div style={analysisErrorHeader}>
            <AlertTriangle size={15} style={{ flexShrink: 0 }} />
            Daten nicht verfügbar
          </div>
          <div>{highlightText(String(item.payload.error), query)}</div>
          <div style={analysisErrorHint}>
            Andere Kennzahlen oder Zeiträume für dieses Unternehmen können trotzdem verfügbar sein.
          </div>
        </div>
      ) : null}

      <div style={miniScoreGrid}>
        <MiniScore label="Erfüllt" value={score.positive} tone="good" />
        <MiniScore label="Kritisch" value={score.negative} tone="bad" />
        <MiniScore label="Neutral" value={score.neutral} />
      </div>

      <div style={criteriaTable}>
        {criteria.length > 0 ? (
          <>
            <CriteriaGroup label="Kritisch" tone="danger" criteria={grouped.critical} query={query} />
            <CriteriaGroup label="Erfüllt" tone="success" criteria={grouped.positive} query={query} />
            <CriteriaGroup label="Neutral" tone="neutral" criteria={grouped.neutral} query={query} />
          </>
        ) : (
          <div style={emptyBox}>Keine Kriterien für diese Kategorie verfügbar.</div>
        )}
      </div>

      {item.payload?.crv ? (
        <div style={crvSection}>
          <div style={crvHeader}>
            <div style={eyebrow}>CRV</div>
            <div style={crvTitle}>Kursziele & Chance-Risiko</div>
          </div>
          <CrvTargetPanel value={item.payload.crv} />
        </div>
      ) : null}
    </div>
  );
}

function MiniScore({ label, value, tone }: { label: string; value: number; tone?: "good" | "bad" }) {
  return (
    <div style={miniScoreBox}>
      <span style={miniScoreLabel}>{label}</span>
      <strong
        style={{
          ...miniScoreValue,
          color: tone === "good" ? theme.colors.successText : tone === "bad" ? theme.colors.dangerText : theme.colors.textPrimary,
        }}
      >
        {value}
      </strong>
    </div>
  );
}

const eyebrow = {
  color: theme.colors.chrome,
  fontSize: "0.82rem",
  fontWeight: 800,
  letterSpacing: "0.04em",
  textTransform: "uppercase" as const,
};

const frequencyPanel = {
  padding: "18px",
  borderRadius: theme.radius.lg,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.borderSubtle}`,
};

const frequencyPanelHeader = {
  display: "flex",
  justifyContent: "space-between",
  gap: "12px",
  alignItems: "center",
  marginBottom: "14px",
};

const frequencyLabel = {
  color: theme.colors.chrome,
  fontSize: "0.76rem",
  fontWeight: 900,
};

const frequencyTitle = {
  color: theme.colors.textPrimary,
  fontSize: "1.05rem",
  fontWeight: 900,
  marginTop: "4px",
};

const analysisMessageBox = {
  padding: "12px 14px",
  borderRadius: theme.radius.md,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
  fontSize: "0.9rem",
  lineHeight: 1.6,
  marginBottom: "12px",
};

const analysisErrorBox = {
  padding: "12px 14px",
  borderRadius: theme.radius.md,
  background: theme.colors.dangerSoft,
  border: `1px solid ${theme.colors.dangerBorder}`,
  color: theme.colors.dangerText,
  fontSize: "0.9rem",
  lineHeight: 1.6,
  marginBottom: "12px",
};

const analysisErrorHeader = {
  display: "flex",
  alignItems: "center",
  gap: "6px",
  fontWeight: 800,
  fontSize: "0.86rem",
  textTransform: "uppercase" as const,
  letterSpacing: "0.02em",
  marginBottom: "6px",
};

const analysisErrorHint = {
  marginTop: "8px",
  color: theme.colors.textSecondary,
  fontSize: "0.86rem",
  lineHeight: 1.6,
};

const miniScoreGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(3, 1fr)",
  gap: "8px",
  marginBottom: "14px",
};

const miniScoreBox = {
  padding: "10px",
  borderRadius: theme.radius.md,
  background: theme.colors.panel,
  border: `1px solid ${theme.colors.borderSubtle}`,
};

const miniScoreLabel = {
  display: "block",
  color: theme.colors.textMuted,
  fontSize: "0.72rem",
  fontWeight: 800,
  marginBottom: "4px",
};

const miniScoreValue = {
  color: theme.colors.textPrimary,
  fontSize: "1rem",
};

const criteriaTable = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "18px",
};

const crvSection = {
  marginTop: "16px",
  paddingTop: "16px",
  borderTop: `1px solid ${theme.colors.borderSubtle}`,
};

const crvHeader = {
  marginBottom: "12px",
};

const crvTitle = {
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  fontWeight: 900,
  marginTop: "4px",
};

const emptyBox = {
  padding: "14px 16px",
  borderRadius: theme.radius.md,
  background: theme.colors.panel,
  border: `1px solid ${theme.colors.borderSubtle}`,
  color: theme.colors.textMuted,
  fontSize: "0.92rem",
  lineHeight: 1.7,
};
