import { useMemo, useState } from "react";
import type { FullResult } from "../../types/analysis";
import { theme } from "../ui/theme";
import { useIsNarrow } from "../../hooks/useMediaQuery";
import { extractCriteria, buildSummary, type AnalysisItem } from "./analysisResultUtils";
import DossierSummaryCard from "./DossierSummaryCard";
import DossierCategoryRail from "./DossierCategoryRail";
import DossierDetailPanel from "./DossierDetailPanel";

type Props = {
  data: FullResult;
  query?: string;
};

function AnalyzeResultsDashboard({ data, query = "" }: Props) {
  const items = useMemo(() => {
    return Object.entries(data.results ?? {}).map(([key, payload]) => {
      const [name, frequency] = key.split("|");

      return {
        key,
        name: name ?? key,
        frequency: frequency ?? "",
        payload,
      };
    });
  }, [data.results]);

  const grouped = useMemo(() => {
    const map: Record<string, AnalysisItem[]> = {};

    for (const item of items) {
      if (!map[item.name]) {
        map[item.name] = [];
      }

      map[item.name].push(item);
    }

    return map;
  }, [items]);

  const groupNames = Object.keys(grouped);
  const [activeGroup, setActiveGroup] = useState(groupNames[0] ?? "");

  const activeItems = grouped[activeGroup] ?? [];

  const summary = useMemo(() => buildSummary(items), [items]);
  const isNarrow = useIsNarrow();

  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const name of groupNames) {
      counts[name] = (grouped[name] ?? []).reduce(
        (sum, item) => sum + extractCriteria(item.payload).length,
        0
      );
    }
    return counts;
  }, [groupNames, grouped]);

  if (items.length === 0) {
    return <div style={emptyBox}>Für diese Analyse wurden keine darstellbaren Ergebnisse gefunden.</div>;
  }

  return (
    <div style={wrapper}>
      <DossierSummaryCard
        symbol={data.symbol}
        bestAssessment={summary.bestAssessment}
        total={items.length}
        positive={summary.positive}
        negative={summary.negative}
        neutral={summary.neutral}
      />

      <div style={{ ...dossierLayout, gridTemplateColumns: isNarrow ? "1fr" : "220px 1fr" }}>
        <aside style={{ ...railColumn, position: isNarrow ? "static" : "sticky" }}>
          <div style={eyebrow}>Kategorien</div>
          <div style={{ marginTop: "12px" }}>
            <DossierCategoryRail
              groupNames={groupNames}
              activeGroup={activeGroup}
              onSelect={setActiveGroup}
              counts={categoryCounts}
            />
          </div>
        </aside>

        <section style={detailColumn}>
          <DossierDetailPanel activeGroup={activeGroup} activeItems={activeItems} query={query} />
        </section>
      </div>
    </div>
  );
}

/* styles */

const wrapper = {
  display: "flex",
  flexDirection: "column" as const,
  gap: theme.spacing(5.5),
};

const eyebrow = {
  color: theme.colors.chrome,
  fontSize: "0.82rem",
  fontWeight: 800,
  letterSpacing: "0.04em",
  textTransform: "uppercase" as const,
};

const dossierLayout = {
  display: "grid",
  gridTemplateColumns: "220px 1fr",
  gap: theme.spacing(5.5),
  alignItems: "start",
};

const railColumn = {
  padding: "18px",
  borderRadius: theme.radius.lg,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.border}`,
  position: "sticky" as const,
  top: "16px",
};

const detailColumn = {
  padding: "24px",
  borderRadius: theme.radius.lg,
  background: theme.colors.panel,
  border: `1px solid ${theme.colors.border}`,
  animation: `analyze-fade-in ${theme.motion.base} ${theme.motion.easing}`,
  minWidth: 0,
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

export default AnalyzeResultsDashboard;
