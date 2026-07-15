import { useEffect, useMemo, useRef, useState } from "react";
import Badge from "../ui/Badge";
import InfoTooltip from "../ui/InfoTooltip";
import { theme } from "../ui/theme";
import { formatMetricValue } from "../metrics/metricFormatting";
import { getMetricConfig } from "../../config/metricsConfig";
import { useIsMobile } from "../../hooks/useMediaQuery";
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
  const isMobile = useIsMobile();
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // Zeigt eine Fade-Kante rechts, solange noch weitere Firmen-Spalten
  // ungescrollt sind — die sticky erste Spalte deckt das linke Ende bereits
  // ab, dort ist kein Hinweis nötig (RESPONSIVE.md R-P1-5).
  const [canScrollRight, setCanScrollRight] = useState(false);

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

  // EVOLVING.md EV-023: Spaltenkopf je Firma um den ISO-Code ergänzen, wenn
  // er von USD abweicht ODER die Firmen in dieser Tabelle unterschiedliche
  // Berichtswährungen haben (dann ist der Hinweis auch bei USD-Firmen
  // hilfreich, um die Abweichung sichtbar zu machen) - im reinen
  // "alle Firmen berichten in USD"-Standardfall bleibt der Spaltenkopf
  // unverändert (keine sichtbare Änderung).
  const companyCurrencies = useMemo(() => {
    const map = new Map<string, string>();
    for (const layer of layers) {
      if (layer.currency && !map.has(layer.groupId)) {
        map.set(layer.groupId, layer.currency);
      }
    }
    return map;
  }, [layers]);
  const distinctCompanyCurrencies = new Set(companyCurrencies.values());
  const hasMixedCompanyCurrencies = distinctCompanyCurrencies.size > 1;

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    function updateFade() {
      setCanScrollRight(el!.scrollLeft + el!.clientWidth < el!.scrollWidth - 2);
    }

    updateFade();
    el.addEventListener("scroll", updateFade, { passive: true });
    const resizeObserver = new ResizeObserver(updateFade);
    resizeObserver.observe(el);
    return () => {
      el.removeEventListener("scroll", updateFade);
      resizeObserver.disconnect();
    };
  }, [rows, groups]);

  if (rows.length === 0 || groups.length === 0) return null;

  const stickyColumnMaxWidth = isMobile ? "140px" : "220px";
  // Mask statt eines farbig überlagerten Divs: blendet nur die
  // Content-Opazität am rechten Rand aus (zeigt den tatsächlichen
  // Hintergrund durch, unabhängig von der Zebra-Streifung der Zeilen) statt
  // eine Farbe erraten zu müssen, die zur wechselnden Zeilenfarbe passt
  // (RESPONSIVE.md R-P1-5).
  const rightFadeMask = canScrollRight
    ? "linear-gradient(to right, black calc(100% - 28px), transparent)"
    : undefined;

  return (
    <div
      ref={scrollRef}
      style={{
        overflowX: "auto",
        WebkitOverflowScrolling: "touch",
        overscrollBehaviorX: "contain",
        borderRadius: theme.radius.lg,
        border: `1px solid ${theme.colors.border}`,
        maskImage: rightFadeMask,
        WebkitMaskImage: rightFadeMask,
      }}
    >
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.88rem" }}>
        <thead>
          <tr>
            <th style={{ ...headerCellStyle, ...stickyColumnStyle, textAlign: "left" }}>Kennzahl</th>
            {groups.map((group) => {
              const currency = companyCurrencies.get(group.groupId);
              const showCurrency = currency && (currency !== "USD" || hasMixedCompanyCurrencies);
              return (
                <th key={group.groupId} style={headerCellStyle}>
                  {group.groupLabel}
                  {showCurrency ? <span style={headerCurrencyStyle}> ({currency})</span> : null}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.metricKey} style={{ background: index % 2 === 0 ? "transparent" : theme.colors.panelAlt }}>
              <td
                style={{
                  ...bodyCellStyle,
                  ...stickyColumnStyle,
                  textAlign: "left",
                  fontWeight: 700,
                  maxWidth: stickyColumnMaxWidth,
                }}
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

/** Wraps a value cell with the "Erfüllt"/"Kritisch" criterion badge - only
 * when `meetsCriterion` is actually set (undefined means no threshold was
 * configured for this metric, in which case no badge is shown at all, not
 * even a neutral one). Mirrors the convention in MetricResultCard.tsx. */
function withCriterionBadge(content: React.ReactNode, meetsCriterion: boolean | undefined) {
  if (meetsCriterion === undefined) return content;
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
      <Badge tone={meetsCriterion ? "success" : "danger"}>{meetsCriterion ? "Erfüllt" : "Kritisch"}</Badge>
      {content}
    </span>
  );
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
    const compact = formatMetricValue(primaryValue, formatKey, layer.currency);
    const fullBreakdown = formatMetricValue(layer.value, layer.metricKey, layer.currency);
    return withCriterionBadge(<span title={fullBreakdown}>{compact}</span>, layer.meetsCriterion);
  }

  const formatted = formatMetricValue(layer.value, layer.metricKey, layer.currency);

  if (typeof layer.value === "string" && layer.value.length > 40) {
    return withCriterionBadge(<span title={layer.value}>{layer.value.slice(0, 40)}…</span>, layer.meetsCriterion);
  }

  return withCriterionBadge(<span>{formatted}</span>, layer.meetsCriterion);
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

const headerCurrencyStyle: React.CSSProperties = {
  color: theme.colors.textMuted,
  fontWeight: 500,
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
