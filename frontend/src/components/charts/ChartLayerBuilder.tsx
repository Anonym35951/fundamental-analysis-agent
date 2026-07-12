import { useEffect, useMemo, useRef, useState } from "react";
import { Trash2 } from "lucide-react";
import { Button, Card, Input, Select, theme } from "../ui";
import { getMetricHistory } from "../../api/customAnalysis";
import type { MetricCatalogEntry } from "../../types/customAnalysis";
import MultiLayerChart from "./MultiLayerChart";
import { LAYER_COLORS, type ChartLayer } from "./chartUtils";

type Layer = {
  id: string;
  symbol: string;
  metricKey: string;
  axis: "left" | "right";
  color: string;
  data: Array<{ date: string; value: number }>;
  isLoading: boolean;
  error: string | null;
};

function medianMagnitude(points: Array<{ value: number }>): number {
  const abs = points.map((p) => Math.abs(p.value)).filter((v) => v > 0);
  if (abs.length === 0) return 0;
  const sorted = [...abs].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

function pickAxisForNewLayer(
  existingLayers: Layer[],
  newSeries: Array<{ value: number }>
): "left" | "right" {
  const newMagnitude = medianMagnitude(newSeries);
  if (newMagnitude === 0) return "left";

  const magnitudeByAxis = (axis: "left" | "right") => {
    const mags = existingLayers
      .filter((layer) => layer.axis === axis && layer.data.length > 0)
      .map((layer) => medianMagnitude(layer.data));
    return mags.length > 0 ? mags.reduce((a, b) => a + b, 0) / mags.length : null;
  };

  const leftMagnitude = magnitudeByAxis("left");
  const rightMagnitude = magnitudeByAxis("right");

  if (leftMagnitude === null && rightMagnitude === null) return "left";

  const ratio = (a: number, b: number) => Math.max(a, b) / Math.min(a, b);

  if (leftMagnitude !== null && rightMagnitude === null) {
    return ratio(newMagnitude, leftMagnitude) <= 20 ? "left" : "right";
  }
  if (rightMagnitude !== null && leftMagnitude === null) {
    return ratio(newMagnitude, rightMagnitude) <= 20 ? "right" : "left";
  }

  // Both axes already occupied — join whichever is the closer order of
  // magnitude rather than always defaulting back to "left".
  return ratio(newMagnitude, leftMagnitude as number) <= ratio(newMagnitude, rightMagnitude as number)
    ? "left"
    : "right";
}

type Props = {
  catalog: MetricCatalogEntry[];
  /** Pre-fills the symbol field for new layers (e.g. the symbol currently
   * shown in the analysis result) — each layer can still be overridden to a
   * different symbol to overlay peers on the same metric. */
  defaultSymbol?: string;
};

/** "Chart-Baukasten": lets a user stack arbitrary metric+symbol combinations
 * as independent layers on one chart, each assignable to a left/right axis.
 * Layers are fetched individually via the synchronous /analyze/custom/history
 * lookup instead of a full analysis job, so adding a peer symbol's metric
 * doesn't require running an analysis for it first. */
export default function ChartLayerBuilder({ catalog, defaultSymbol }: Props) {
  const timeseriesCatalog = useMemo(
    () => catalog.filter((entry) => entry.result_shape === "timeseries"),
    [catalog]
  );
  const catalogByKey = useMemo(() => new Map(catalog.map((entry) => [entry.key, entry])), [catalog]);

  const [layers, setLayers] = useState<Layer[]>([]);
  const [draftSymbol, setDraftSymbol] = useState(defaultSymbol ?? "");
  const [draftMetricKey, setDraftMetricKey] = useState("");

  // Seeds the field from the page's current symbol exactly once, the first
  // time it becomes available — NOT on every render where the field happens
  // to be empty, otherwise clearing the field to type a different symbol
  // would get immediately overwritten again.
  const hasSeededSymbol = useRef(false);
  useEffect(() => {
    if (!hasSeededSymbol.current && defaultSymbol) {
      setDraftSymbol(defaultSymbol);
      hasSeededSymbol.current = true;
    }
  }, [defaultSymbol]);

  useEffect(() => {
    if (!draftMetricKey && timeseriesCatalog.length > 0) {
      setDraftMetricKey(timeseriesCatalog[0].key);
    }
  }, [draftMetricKey, timeseriesCatalog]);

  async function handleAddLayer() {
    const symbol = draftSymbol.trim().toUpperCase();
    if (!symbol || !draftMetricKey) return;

    const id = `${symbol}:${draftMetricKey}:${Date.now()}`;
    const color = LAYER_COLORS[layers.length % LAYER_COLORS.length];

    setLayers((prev) => [
      ...prev,
      { id, symbol, metricKey: draftMetricKey, axis: "left", color, data: [], isLoading: true, error: null },
    ]);

    try {
      const response = await getMetricHistory({ key: draftMetricKey, symbol });
      const series = response.series ?? [];
      setLayers((prev) => {
        // Pick whichever axis already holds values of a comparable order of
        // magnitude — without this, e.g. an EV/EBITDA ratio (~10-100) added
        // after a market-cap layer (~10^11) defaults to the same axis and
        // renders as a flat line pinned near zero on that huge scale.
        const axis =
          series.length > 0 ? pickAxisForNewLayer(prev.filter((layer) => layer.id !== id), series) : "left";

        return prev.map((layer) =>
          layer.id === id
            ? {
                ...layer,
                axis,
                data: series,
                isLoading: false,
                error: response.error ?? (series.length === 0 ? "Keine Zeitreihe verfügbar." : null),
              }
            : layer
        );
      });
    } catch (error) {
      setLayers((prev) =>
        prev.map((layer) =>
          layer.id === id
            ? { ...layer, isLoading: false, error: error instanceof Error ? error.message : "Fehler beim Laden." }
            : layer
        )
      );
    }
  }

  function removeLayer(id: string) {
    setLayers((prev) => prev.filter((layer) => layer.id !== id));
  }

  function toggleAxis(id: string) {
    setLayers((prev) =>
      prev.map((layer) =>
        layer.id === id ? { ...layer, axis: layer.axis === "left" ? "right" : "left" } : layer
      )
    );
  }

  const chartLayers: ChartLayer[] = layers
    .filter((layer) => layer.data.length > 0)
    .map((layer) => ({
      id: layer.id,
      label: `${layer.symbol} · ${catalogByKey.get(layer.metricKey)?.label ?? layer.metricKey}`,
      data: layer.data,
      axis: layer.axis,
      color: layer.color,
    }));

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
      <MultiLayerChart layers={chartLayers} />

      {layers.length > 0 ? (
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          {layers.map((layer) => (
            <div key={layer.id} style={layerRow}>
              <span style={{ ...colorSwatch, background: layer.color }} />
              <span style={layerLabel}>
                {layer.symbol} · {catalogByKey.get(layer.metricKey)?.label ?? layer.metricKey}
              </span>
              {layer.isLoading ? (
                <span style={layerStatus}>Wird geladen...</span>
              ) : layer.error ? (
                <span style={{ ...layerStatus, color: theme.colors.dangerText }}>{layer.error}</span>
              ) : null}
              <Button variant="ghost" onClick={() => toggleAxis(layer.id)} style={axisButton}>
                Achse: {layer.axis === "left" ? "Links" : "Rechts"}
              </Button>
              <button type="button" onClick={() => removeLayer(layer.id)} aria-label="Layer entfernen" style={removeButton}>
                <Trash2 size={14} color={theme.colors.textMuted} />
              </button>
            </div>
          ))}
        </div>
      ) : null}

      <Card variant="alt">
        <div style={{ display: "flex", gap: "10px", alignItems: "flex-end", flexWrap: "wrap" }}>
          <div style={{ flex: "0 1 160px" }}>
            <label style={fieldLabel}>Symbol</label>
            <Input
              value={draftSymbol}
              onChange={(e) => setDraftSymbol(e.target.value.toUpperCase())}
              placeholder="z. B. AAPL"
            />
          </div>
          <div style={{ flex: "1 1 220px" }}>
            <label style={fieldLabel}>Kennzahl</label>
            <Select value={draftMetricKey} onChange={(e) => setDraftMetricKey(e.target.value)}>
              {timeseriesCatalog.map((entry) => (
                <option key={entry.key} value={entry.key}>
                  {entry.label}
                </option>
              ))}
            </Select>
          </div>
          <Button onClick={handleAddLayer} disabled={!draftSymbol.trim() || !draftMetricKey}>
            Layer hinzufügen
          </Button>
        </div>
      </Card>
    </div>
  );
}

const layerRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "10px",
  padding: "8px 12px",
  borderRadius: theme.radius.md,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
};

const colorSwatch: React.CSSProperties = {
  width: "10px",
  height: "10px",
  borderRadius: theme.radius.pill,
  flexShrink: 0,
};

const layerLabel: React.CSSProperties = {
  flex: 1,
  fontSize: "0.88rem",
  fontWeight: 700,
  color: theme.colors.textPrimary,
};

const layerStatus: React.CSSProperties = {
  fontSize: "0.78rem",
  color: theme.colors.textMuted,
};

const axisButton: React.CSSProperties = {
  padding: "6px 12px",
  fontSize: "0.78rem",
};

const removeButton: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  border: "none",
  background: "transparent",
  cursor: "pointer",
  padding: "4px",
};

const fieldLabel: React.CSSProperties = {
  display: "block",
  marginBottom: "8px",
  color: theme.colors.textSecondary,
  fontWeight: 600,
  fontSize: "0.88rem",
};
