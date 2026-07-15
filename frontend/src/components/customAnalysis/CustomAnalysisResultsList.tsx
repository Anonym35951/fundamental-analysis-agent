import { useMemo, useState } from "react";
import { Card, theme } from "../ui";
import CrvTargetPanel from "../metrics/CrvTargetPanel";
import MultiLayerChart from "../charts/MultiLayerChart";
import TimeRangeFilter from "../charts/TimeRangeFilter";
import PercentChangeBadge from "../charts/PercentChangeBadge";
import { computePercentChange, filterSeriesByRange, isPercentChangeEligibleUnit, type TimeRange } from "../charts/chartUtils";
import ComparePivotTable from "../compare/ComparePivotTable";
import { getCompanyColor, mapCompanyComplexMetrics, mapCompanyMetricsToLayers } from "../../compare/mapping";
import { useLivePrice } from "../../hooks/useLivePrice";
import { getMetricConfig } from "../../config/metricsConfig";
import type { CompareGroupMeta, CompareLayer } from "../../types/compare";
import type { CustomAnalysisResult, MetricCatalogEntry } from "../../types/customAnalysis";

// EVOLVING.md EV-041: identische Range-Teilmenge wie ComparePage - 1M-6M
// sind bei jaehrlichen/quartalsweisen Fundamentaldaten fachlich sinnlos.
const FUNDAMENTAL_RANGE_OPTIONS: TimeRange[] = ["1y", "2y", "5y", "max"];

type Props = {
  catalog: MetricCatalogEntry[];
  result: CustomAnalysisResult;
};

/** Renders a finished custom-analysis result the same way ComparePage renders
 * a comparison: a price row + scalar metrics in a pivot table, and one
 * full-size chart per historical (timeseries) metric — instead of a card
 * list with a redundant "—" table row alongside its own chart. Reuses the
 * exact same mapping helpers as ComparePage so a single symbol's result
 * looks identical to a 1-company comparison. */
export default function CustomAnalysisResultsList({ catalog, result }: Props) {
  const { price, error: priceError } = useLivePrice(result.symbol);
  // Zeitraum je Chart-Sektion, Schlüssel = layer.id (EVOLVING.md EV-041,
  // D7: React-State pro Chart, keine URL-Persistenz).
  const [timeRanges, setTimeRanges] = useState<Record<string, TimeRange>>({});

  const layers = useMemo<CompareLayer[]>(
    () =>
      mapCompanyMetricsToLayers(
        result.symbol,
        result.metrics,
        catalog,
        getCompanyColor(0),
        result.reporting_currency
      ),
    [result.symbol, result.metrics, catalog, result.reporting_currency]
  );

  const complexResults = useMemo(
    () => mapCompanyComplexMetrics(result.symbol, result.metrics, catalog),
    [result.symbol, result.metrics, catalog]
  );

  const chartLayers = layers.filter((layer) => layer.chartEligible && layer.data);

  const priceLayer: CompareLayer = {
    id: `${result.symbol}:__price`,
    groupId: result.symbol,
    groupLabel: result.symbol,
    metricKey: "__price",
    label: "Aktienkurs",
    axis: "left",
    color: getCompanyColor(0),
    chartEligible: false,
    value: price ?? null,
    error: priceError,
    // Kursbasiert, immer USD (NYSE/NASDAQ-Universum) - unabhängig von der
    // Berichtswährung der Fundamentaldaten (EVOLVING.md EV-022).
    currency: "USD",
  };

  const tableLayers = [priceLayer, ...layers.filter((layer) => !layer.chartEligible)];
  const groups: CompareGroupMeta[] = [{ groupId: result.symbol, groupLabel: result.symbol }];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
      {chartLayers.map((layer) => {
        const range = timeRanges[layer.id] ?? "max";
        const filteredData = filterSeriesByRange(layer.data ?? [], range);
        // EVOLVING.md EV-023/EV-051: dieselbe Bedingung für Currency-Label
        // und %-Badge - Ratio-/Margen-Charts bekommen keins von beidem.
        const isEligibleUnit = isPercentChangeEligibleUnit(getMetricConfig(layer.metricKey)?.unit);

        return (
          <section key={layer.id} style={chartSection}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px", marginBottom: "10px" }}>
              <div style={{ ...chartSectionEyebrow, marginBottom: 0 }}>{layer.label}</div>
              <TimeRangeFilter
                value={range}
                onChange={(next) => setTimeRanges((prev) => ({ ...prev, [layer.id]: next }))}
                options={FUNDAMENTAL_RANGE_OPTIONS}
              />
            </div>
            {isEligibleUnit ? (
              <div style={percentBadgeRowStyle}>
                <PercentChangeBadge result={computePercentChange(filteredData)} />
              </div>
            ) : null}
            {filteredData.length >= 2 ? (
              <MultiLayerChart
                layers={[
                  {
                    id: layer.id,
                    label: layer.label,
                    data: filteredData,
                    axis: "left",
                    color: layer.color,
                    // EVOLVING.md EV-023: nur für unit==="currency"-Kennzahlen,
                    // Ratio-/Margen-Charts bekommen bewusst keine Währungslabel.
                    currency: isEligibleUnit ? layer.currency : undefined,
                  },
                ]}
                height={300}
              />
            ) : (
              <div style={emptyRangeStyle}>Für diesen Zeitraum liegen zu wenige Datenpunkte vor – Zeitraum vergrößern.</div>
            )}
          </section>
        );
      })}

      {complexResults.map((complex) => (
        <Card key={complex.metricKey} variant="glass">
          <h3 style={{ margin: "0 0 12px 0", color: theme.colors.textPrimary }}>{complex.label}</h3>
          {complex.error ? (
            <p style={{ color: theme.colors.dangerText, margin: 0 }}>{complex.error}</p>
          ) : (
            <CrvTargetPanel value={complex.value} />
          )}
        </Card>
      ))}

      {tableLayers.length > 0 ? <ComparePivotTable layers={tableLayers} groups={groups} /> : null}
    </div>
  );
}

const chartSection: React.CSSProperties = {
  background: theme.colors.panelAlt,
  borderRadius: theme.radius.lg,
  padding: "20px 22px",
  border: `1px solid ${theme.colors.borderSubtle}`,
};

const emptyRangeStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  height: "160px",
  color: theme.colors.textMuted,
  fontSize: "0.92rem",
  border: `1px dashed ${theme.colors.borderSubtle}`,
  borderRadius: theme.radius.md,
};

const percentBadgeRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: "8px",
  marginBottom: "10px",
};

const chartSectionEyebrow: React.CSSProperties = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase",
  marginBottom: "10px",
};
