import { useMemo } from "react";
import { Card, theme } from "../ui";
import CrvTargetPanel from "../metrics/CrvTargetPanel";
import MultiLayerChart from "../charts/MultiLayerChart";
import ComparePivotTable from "../compare/ComparePivotTable";
import { getCompanyColor, mapCompanyComplexMetrics, mapCompanyMetricsToLayers } from "../../compare/mapping";
import { useLivePrice } from "../../hooks/useLivePrice";
import type { CompareGroupMeta, CompareLayer } from "../../types/compare";
import type { CustomAnalysisResult, MetricCatalogEntry } from "../../types/customAnalysis";

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

  const layers = useMemo<CompareLayer[]>(
    () => mapCompanyMetricsToLayers(result.symbol, result.metrics, catalog, getCompanyColor(0)),
    [result.symbol, result.metrics, catalog]
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
  };

  const tableLayers = [priceLayer, ...layers.filter((layer) => !layer.chartEligible)];
  const groups: CompareGroupMeta[] = [{ groupId: result.symbol, groupLabel: result.symbol }];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
      {chartLayers.map((layer) => (
        <section key={layer.id} style={chartSection}>
          <div style={chartSectionEyebrow}>{layer.label}</div>
          <MultiLayerChart
            layers={[{ id: layer.id, label: layer.label, data: layer.data ?? [], axis: "left", color: layer.color }]}
            height={300}
          />
        </section>
      ))}

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

const chartSectionEyebrow: React.CSSProperties = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase",
  marginBottom: "10px",
};
