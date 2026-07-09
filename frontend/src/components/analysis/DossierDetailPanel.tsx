import StackedCards from "../ui/StackedCards";
import { theme } from "../ui/theme";
import { highlightText } from "../metrics/metricFormatting";
import type { AnalysisItem } from "./analysisResultUtils";
import AnalysisFrequencyPanel from "./AnalysisFrequencyPanel";

type Props = {
  activeGroup: string;
  activeItems: AnalysisItem[];
  query?: string;
};

export default function DossierDetailPanel({ activeGroup, activeItems, query }: Props) {
  const activePrimaryItem = activeItems[0];
  if (!activePrimaryItem) return null;

  return (
    <div key={activeGroup}>
      <div style={detailHeader}>
        <div>
          <div style={eyebrow}>Ausgewählte Kategorie</div>
          <h3 style={sectionTitle}>{activeGroup}</h3>
        </div>

        <div style={frequencyPills}>
          {activeItems.map((item) => (
            <span key={item.key} style={frequencyPill}>
              {item.frequency.toUpperCase()}
            </span>
          ))}
        </div>
      </div>

      <div style={assessmentBox}>
        <div style={assessmentLabel}>Kurzfazit</div>
        <div style={assessmentValue}>{activePrimaryItem.payload?.overall_assessment ?? "Keine Einschätzung verfügbar"}</div>

        {activePrimaryItem.payload?.message ? (
          <p style={assessmentMessage}>{highlightText(String(activePrimaryItem.payload.message), query)}</p>
        ) : null}
      </div>

      <StackedCards
        items={activeItems}
        getKey={(item) => item.key}
        maxPeek={2}
        expandLabel={(hidden) => `${hidden} weitere Zeiträume anzeigen`}
        renderCard={(item) => <AnalysisFrequencyPanel item={item} query={query} />}
      />
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

const sectionTitle = {
  margin: "8px 0 0 0",
  color: theme.colors.textPrimary,
  fontSize: "1.45rem",
  lineHeight: 1.2,
  letterSpacing: "-0.03em",
};

const detailHeader = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "flex-start",
  gap: "16px",
  marginBottom: "18px",
};

const frequencyPills = {
  display: "flex",
  gap: "8px",
  flexWrap: "wrap" as const,
};

const frequencyPill = {
  padding: "8px 10px",
  borderRadius: theme.radius.pill,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
  fontSize: "0.78rem",
  fontWeight: 900,
};

const assessmentBox = {
  padding: "16px 18px",
  borderRadius: theme.radius.md,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  marginBottom: "18px",
};

const assessmentLabel = {
  color: theme.colors.chrome,
  fontSize: "0.82rem",
  fontWeight: 900,
  textTransform: "uppercase" as const,
  marginBottom: "8px",
};

const assessmentValue = {
  color: theme.colors.textPrimary,
  fontSize: "1.18rem",
  fontWeight: 900,
};

const assessmentMessage = {
  margin: "10px 0 0 0",
  color: theme.colors.textSecondary,
  fontSize: "0.95rem",
  lineHeight: 1.7,
};
