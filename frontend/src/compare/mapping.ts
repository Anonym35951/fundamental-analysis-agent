import type { CustomMetricResult, MetricCatalogEntry } from "../types/customAnalysis";
import type { CompareLayer } from "../types/compare";
import { LAYER_COLORS } from "../components/charts/chartUtils";

/** Each company gets one stable color (by its index among the companies
 * currently rendered) that stays the same across every metric's chart box —
 * since each metric now gets its own single-axis chart, the only thing that
 * needs distinguishing within a box is the company, not the metric. */
export function getCompanyColor(index: number): string {
  return LAYER_COLORS[index % LAYER_COLORS.length];
}

function shouldSkipCatalogEntry(catalogEntry: MetricCatalogEntry | undefined): boolean {
  return catalogEntry?.result_shape === "complex";
}

/** Maps one company's custom-analysis job result into CompareLayers — one
 * layer per selected metric, namespacing-free since every company shares
 * the same metric selection. `color` is the company's stable color (see
 * `getCompanyColor`), applied to every layer for this company so it stays
 * recognizable across all of that company's metric chart boxes. */
export function mapCompanyMetricsToLayers(
  symbol: string,
  metrics: Record<string, CustomMetricResult>,
  catalog: MetricCatalogEntry[],
  color: string,
  reportingCurrency?: string | null
): CompareLayer[] {
  const catalogByKey = new Map(catalog.map((entry) => [entry.key, entry]));

  const entries = Object.entries(metrics).filter(([key]) => !shouldSkipCatalogEntry(catalogByKey.get(key)));

  return entries.map(([key, metric]) => {
    const series = metric.series ?? [];
    return {
      id: `${symbol}:${key}`,
      groupId: symbol,
      groupLabel: symbol,
      metricKey: key,
      label: catalogByKey.get(key)?.label ?? key,
      axis: "left",
      color,
      chartEligible: series.length > 0,
      data: series.length > 0 ? series : undefined,
      value: metric.value,
      error: metric.error ?? null,
      meetsCriterion: metric.meets_criterion,
      currency: reportingCurrency,
    };
  });
}

export type CompareComplexResult = {
  metricKey: string;
  label: string;
  symbol: string;
  value: unknown;
  error: string | null;
};

/** Counterpart to mapCompanyMetricsToLayers for result_shape "complex"
 * metrics (TBV-/EBIT-Bandbreite, CRV, Kursziel Price-/EV-Multiples) — these
 * don't fit the pivot-table/line-chart shape CompareLayer assumes, so they
 * get a separate result type rendered via CrvTargetPanel instead. */
export function mapCompanyComplexMetrics(
  symbol: string,
  metrics: Record<string, CustomMetricResult>,
  catalog: MetricCatalogEntry[]
): CompareComplexResult[] {
  const catalogByKey = new Map(catalog.map((entry) => [entry.key, entry]));

  return Object.entries(metrics)
    .filter(([key]) => catalogByKey.get(key)?.result_shape === "complex")
    .map(([key, metric]) => ({
      metricKey: key,
      label: catalogByKey.get(key)?.label ?? key,
      symbol,
      value: metric.value,
      error: metric.error ?? null,
    }));
}
