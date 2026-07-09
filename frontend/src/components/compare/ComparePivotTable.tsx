import { useMemo } from "react";
import Badge from "../ui/Badge";
import InfoTooltip from "../ui/InfoTooltip";
import { theme } from "../ui/theme";
import { formatMetricValue } from "../metrics/metricFormatting";
import { getMetricConfig } from "../../config/metricsConfig";
import type { CompareGroupMeta, CompareLayer } from "../../types/compare";

type Props = {
  layers: CompareLayer[];
  groups: CompareGroupMeta[];
};

type Row = {
  metricKey: string;
  label: string;
};

/** One row per selected metric, one column per company — every company
 * shares the same metric selection, so rows line up directly across
 * columns with no namespacing needed. */
export default function ComparePivotTable({ layers, groups }: Props) {
  const rows = useMemo<Row[]>(() => {
    const seen = new Map<string, string>();
    for (const layer of layers) {
      if (!seen.has(layer.metricKey)) {
        seen.set(layer.metricKey, layer.label);
      }
    }
    return Array.from(seen.entries()).map(([metricKey, label]) => ({ metricKey, label }));
  }, [layers]);

  const cellsByMetric = useMemo(() => {
    const map = new Map<string, Map<string, CompareLayer>>();
    for (const layer of layers) {
      if (!map.has(layer.metricKey)) {
        map.set(layer.metricKey, new Map());
      }
      map.get(layer.metricKey)!.set(layer.groupId, layer);
    }
    return map;
  }, [layers]);

  if (rows.length === 0 || groups.length === 0) return null;

  return (
    <div style={{ overflowX: "auto", borderRadius: theme.radius.lg, border: `1px solid ${theme.colors.border}` }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem" }}>
        <thead>
          <tr>
            <th style={{ ...headerCellStyle, ...stickyColumnStyle, textAlign: "left" }}>Kennzahl</th>
            {groups.map((group) => (
              <th key={group.groupId} style={headerCellStyle}>
                {group.groupLabel}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.metricKey} style={{ background: index % 2 === 0 ? "transparent" : theme.colors.panelAlt }}>
              <td
                style={{ ...bodyCellStyle, ...stickyColumnStyle, textAlign: "left", fontWeight: 700, maxWidth: "220px" }}
                title={row.label}
              >
                <span style={{ display: "inline-flex", alignItems: "center", gap: "5px", maxWidth: "100%" }}>
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>{row.label}</span>
                  <InfoTooltip metricKey={row.metricKey} />
                </span>
              </td>
              {groups.map((group) => {
                const layer = cellsByMetric.get(row.metricKey)?.get(group.groupId);
                return (
                  <td key={group.groupId} style={bodyCellStyle}>
                    <Cell layer={layer} />
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Picks the single most representative field out of a multi-field metric
 * object (e.g. the inflation-rate metric returns CPI/date bookkeeping
 * alongside the actual rate) so the table cell can show one compact value
 * instead of every field strung together — that's what was blowing up the
 * column width. The full breakdown is still reachable via the cell's
 * tooltip. */
function pickPrimaryObjectValue(value: Record<string, unknown>, metricKey: string): [string, unknown] {
  const entries = Object.entries(value);

  if (metricKey === "get_current_tbv_and_price" && "price_to_tbv" in value) {
    return ["price_to_tbv", value.price_to_tbv];
  }

  const keyMatch = entries.find(([objectKey]) => metricKey.includes(objectKey) || objectKey.includes("rate"));
  if (keyMatch) return keyMatch;

  const numericMatch = entries.find(([, objectValue]) => typeof objectValue === "number");
  return numericMatch ?? entries[0];
}

function Cell({ layer }: { layer: CompareLayer | undefined }) {
  if (!layer) {
    return <span style={{ color: theme.colors.textMuted }}>—</span>;
  }

  if (layer.error) {
    return (
      <span style={{ color: theme.colors.dangerText }} title={layer.error}>
        {layer.error.length > 40 ? `${layer.error.slice(0, 40)}…` : layer.error}
      </span>
    );
  }

  if (typeof layer.value === "boolean") {
    return <Badge tone={layer.value ? "success" : "danger"}>{layer.value ? "Ja" : "Nein"}</Badge>;
  }

  if (layer.value !== null && typeof layer.value === "object" && !Array.isArray(layer.value)) {
    const [pickedKey, primaryValue] = pickPrimaryObjectValue(layer.value as Record<string, unknown>, layer.metricKey);
    // Prefer the picked sub-field's own unit config when it has one (e.g.
    // price_to_tbv is a unitless ratio, distinct from its sibling currency
    // fields) — otherwise fall back to the outer metric key so cases like
    // annual_inflation_rate's "%" unit, keyed by the catalog metric rather
    // than the dict's internal field name, still apply.
    const formatKey = getMetricConfig(pickedKey) ? pickedKey : layer.metricKey;
    const compact = formatMetricValue(primaryValue, formatKey);
    const fullBreakdown = formatMetricValue(layer.value, layer.metricKey);
    return <span title={fullBreakdown}>{compact}</span>;
  }

  const formatted = formatMetricValue(layer.value, layer.metricKey);

  if (typeof layer.value === "string" && layer.value.length > 40) {
    return <span title={layer.value}>{layer.value.slice(0, 40)}…</span>;
  }

  return <span>{formatted}</span>;
}

const headerCellStyle: React.CSSProperties = {
  padding: "8px 10px",
  textAlign: "center",
  color: theme.colors.textSecondary,
  fontWeight: 700,
  fontSize: "0.8rem",
  borderBottom: `1px solid ${theme.colors.border}`,
  whiteSpace: "nowrap",
};

const bodyCellStyle: React.CSSProperties = {
  padding: "8px 10px",
  textAlign: "center",
  color: theme.colors.textPrimary,
  whiteSpace: "nowrap",
  maxWidth: "180px",
  overflow: "hidden",
  textOverflow: "ellipsis",
};

const stickyColumnStyle: React.CSSProperties = {
  position: "sticky",
  left: 0,
  background: theme.colors.panel,
  zIndex: 1,
};
