import { useEffect, useMemo, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { Card, Input, theme } from "../ui";
import type { CriterionOperator, MetricCatalogEntry, MetricSelection } from "../../types/customAnalysis";
import MetricPickerCard from "./MetricPickerCard";

/** Groups of metric keys that are mutually exclusive — selecting one
 * deselects the others in the same group. Extend this list for future
 * either/or metric pairs without touching the selection logic itself. */
const MUTUALLY_EXCLUSIVE_GROUPS: string[][] = [];

function exclusiveGroupFor(key: string): string[] | undefined {
  return MUTUALLY_EXCLUSIVE_GROUPS.find((group) => group.includes(key));
}

type Props = {
  catalog: MetricCatalogEntry[];
  isLoadingCatalog: boolean;
  initialMetrics?: MetricSelection[];
  onChange: (metrics: MetricSelection[]) => void;
  /** Skips the threshold-criterion UI on every tile — used by the Compare
   * workspace, which only ever shows raw values, never pass/fail badges. */
  hideCriterion?: boolean;
};

/** Metric search + grouped picker + per-metric param/criterion form,
 * extracted from DefinitionBuilder so the same picker UI can drive both the
 * "save a definition" flow and the ad-hoc (run-once, no save) flow without
 * duplicating the selection logic. */
export default function MetricCatalogPicker({
  catalog,
  isLoadingCatalog,
  initialMetrics = [],
  onChange,
  hideCriterion = false,
}: Props) {
  const [selections, setSelections] = useState<Map<string, MetricSelection>>(
    () => new Map(initialMetrics.map((m) => [m.key, m]))
  );
  const [query, setQuery] = useState("");
  // Which categories the user has manually toggled open — independent of
  // each other (multiple can be open at once). Starts empty: every category
  // begins collapsed, even if it already holds a selection.
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(() => new Set());

  useEffect(() => {
    onChange(Array.from(selections.values()));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selections]);

  const catalogByKey = useMemo(() => new Map(catalog.map((entry) => [entry.key, entry])), [catalog]);

  const groups = useMemo(() => {
    const filtered = query.trim()
      ? catalog.filter(
          (entry) =>
            entry.label.toLowerCase().includes(query.trim().toLowerCase()) ||
            entry.category.toLowerCase().includes(query.trim().toLowerCase())
        )
      : catalog;

    const map = new Map<string, MetricCatalogEntry[]>();
    for (const entry of filtered) {
      const list = map.get(entry.category) ?? [];
      list.push(entry);
      map.set(entry.category, list);
    }
    return map;
  }, [catalog, query]);

  const categoryCounts = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const [category, entries] of groups) {
      counts[category] = entries.length;
    }
    return counts;
  }, [groups]);

  function toggleCategory(category: string) {
    setExpandedCategories((prev) => {
      const next = new Set(prev);
      if (next.has(category)) {
        next.delete(category);
      } else {
        next.add(category);
      }
      return next;
    });
  }

  function toggleMetric(entry: MetricCatalogEntry) {
    setSelections((prev) => {
      const next = new Map(prev);
      if (next.has(entry.key)) {
        next.delete(entry.key);
      } else {
        const exclusiveGroup = exclusiveGroupFor(entry.key);
        if (exclusiveGroup) {
          for (const otherKey of exclusiveGroup) {
            if (otherKey !== entry.key) next.delete(otherKey);
          }
        }

        const defaultParams: Record<string, unknown> = {};
        for (const param of entry.params) {
          if (param.default !== null && param.default !== undefined) {
            defaultParams[param.name] = param.default;
          }
        }
        next.set(entry.key, { key: entry.key, params: defaultParams });
      }
      return next;
    });
  }

  function updateParam(key: string, paramName: string, value: string | number | undefined) {
    setSelections((prev) => {
      const next = new Map(prev);
      const current = next.get(key);
      if (!current) return prev;
      next.set(key, { ...current, params: { ...current.params, [paramName]: value } });
      return next;
    });
  }

  function updateCriterion(key: string, operator: CriterionOperator | "", threshold: string) {
    setSelections((prev) => {
      const next = new Map(prev);
      const current = next.get(key);
      if (!current) return prev;
      if (!operator || threshold === "") {
        next.set(key, { ...current, criterion: null });
        return next;
      }
      next.set(key, { ...current, criterion: { operator, threshold: Number(threshold) } });
      return next;
    });
  }

  const selectionList = Array.from(selections.values());
  const isSearching = query.trim().length > 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
      <Card variant="glass">
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: "14px",
            gap: "12px",
          }}
        >
          <h3 style={{ margin: 0, color: theme.colors.textPrimary }}>
            Metriken auswählen ({selectionList.length})
          </h3>
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Suche..."
            style={{ maxWidth: "240px", flex: "1 1 200px" }}
          />
        </div>

        {isLoadingCatalog ? (
          <p style={{ color: theme.colors.textSecondary }}>Katalog wird geladen...</p>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
            {Array.from(groups.entries()).map(([category, entries]) => {
              const selectedCount = entries.filter((entry) => selections.has(entry.key)).length;
              const isOpen = expandedCategories.has(category) || isSearching;

              return (
                <div
                  key={category}
                  style={{
                    borderRadius: theme.radius.lg,
                    border: `1px solid ${isOpen ? theme.colors.chromeBorder : theme.colors.border}`,
                    background: theme.colors.panelAlt,
                    overflow: "hidden",
                    transition: `border-color ${theme.motion.base} ${theme.motion.easing}`,
                  }}
                >
                  <button
                    type="button"
                    onClick={() => toggleCategory(category)}
                    style={{
                      width: "100%",
                      minHeight: "48px",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      flexWrap: "wrap",
                      gap: "8px 12px",
                      padding: "13px 16px",
                      background: isOpen ? theme.colors.chromeSoft : "transparent",
                      border: "none",
                      cursor: "pointer",
                      color: theme.colors.textPrimary,
                      transition: `background ${theme.motion.base} ${theme.motion.easing}`,
                    }}
                  >
                    <span style={{ display: "flex", alignItems: "center", gap: "10px", flexWrap: "wrap", minWidth: 0 }}>
                      <span
                        style={{
                          fontWeight: 800,
                          fontSize: "0.92rem",
                        }}
                      >
                        {category}
                      </span>
                      <span style={{ fontSize: "0.74rem", fontWeight: 800, color: theme.colors.textMuted }}>
                        {categoryCounts[category] ?? 0}
                      </span>
                      {selectedCount > 0 ? (
                        <span
                          style={{
                            fontSize: "0.72rem",
                            fontWeight: 700,
                            padding: "2px 8px",
                            borderRadius: theme.radius.pill,
                            background: theme.colors.successSoft,
                            border: `1px solid ${theme.colors.successBorder}`,
                            color: theme.colors.success,
                            whiteSpace: "nowrap",
                          }}
                        >
                          {selectedCount} ausgewählt
                        </span>
                      ) : null}
                    </span>
                    <ChevronDown
                      size={16}
                      style={{
                        flexShrink: 0,
                        transform: isOpen ? "rotate(180deg)" : "rotate(0deg)",
                        transition: `transform ${theme.motion.base} ${theme.motion.easing}`,
                        color: theme.colors.textMuted,
                      }}
                    />
                  </button>

                  <AnimatePresence initial={false}>
                    {isOpen ? (
                      <motion.div
                        key="content"
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
                        style={{ overflow: "hidden" }}
                      >
                        <div
                          style={{
                            display: "grid",
                            // min() caps the track minimum at the available
                            // width — plain minmax(240px, 1fr) would force a
                            // 240px-wide track even inside a narrower mobile
                            // accordion panel, pushing tiles past the edge
                            // instead of collapsing to one full-width column.
                            gridTemplateColumns: "repeat(auto-fit, minmax(min(240px, 100%), 1fr))",
                            gap: "10px",
                            padding: "12px 16px 16px",
                          }}
                        >
                          {entries.map((entry) => (
                            <MetricPickerCard
                              key={entry.key}
                              entry={entry}
                              selection={selections.get(entry.key)}
                              isSelected={selections.has(entry.key)}
                              onToggle={toggleMetric}
                              onParamChange={updateParam}
                              onCriterionChange={updateCriterion}
                              hideCriterion={hideCriterion}
                            />
                          ))}
                        </div>
                      </motion.div>
                    ) : null}
                  </AnimatePresence>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      {selectionList.length > 0 ? (
        <Card variant="alt">
          <div style={{ fontSize: "0.82rem", color: theme.colors.textMuted, fontWeight: 700, marginBottom: "8px", textTransform: "uppercase" }}>
            Ausgewählt
          </div>
          <div style={{ color: theme.colors.textSecondary, fontSize: "0.92rem" }}>
            {selectionList.map((s) => catalogByKey.get(s.key)?.label ?? s.key).join(", ")}
          </div>
        </Card>
      ) : null}
    </div>
  );
}
