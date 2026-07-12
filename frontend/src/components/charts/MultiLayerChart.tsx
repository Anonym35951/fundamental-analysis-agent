import { useCallback, useMemo, useRef, useState } from "react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { theme, useChartTokens } from "../ui/theme";
import { useIsMobile } from "../../hooks/useMediaQuery";

export type ChartLayer = {
  id: string;
  label: string;
  data: Array<{ date: string; value: number }>;
  axis: "left" | "right";
  color: string;
};

type Props = {
  layers: ChartLayer[];
  height?: number;
};

/** Deliberate exception to the app's otherwise-monochrome theme: the compare
 * chart can overlay many company×metric series at once, so it needs a large,
 * highly saturated, mutually-distinguishable categorical palette instead of
 * the 6-shade greyscale ramp used elsewhere. Colors alternate warm/cool/hue
 * families so consecutive indices (sequential assignment in
 * compare/mapping.ts) never land next to a visually similar neighbor. Cycles
 * only once a comparison exceeds this many series. */
export const LAYER_COLORS = [
  "#22d3ee", // cyan
  "#fb7185", // rose
  "#a3e635", // lime
  "#c084fc", // violet
  "#fbbf24", // amber
  "#38bdf8", // sky
  "#f472b6", // pink
  "#4ade80", // green
  "#fb923c", // orange
  "#818cf8", // indigo
  "#facc15", // yellow
  "#2dd4bf", // teal
  "#f87171", // red
  "#a78bfa", // purple
  "#34d399", // emerald
  "#e879f9", // fuchsia
  "#60a5fa", // blue
  "#fde047", // pale yellow
  "#d946ef", // magenta
  "#5eead4", // light teal
];

/** Compact German number formatting (k / Mio. / Mrd.) so large axis ticks
 * and tooltip values (market cap, enterprise value, ...) stay readable
 * instead of rendering as long unbroken digit strings. */
export function formatCompactNumber(value: number): string {
  const abs = Math.abs(value);

  if (abs >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(abs >= 10_000_000_000 ? 0 : 1)} Mrd.`;
  }
  if (abs >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(abs >= 10_000_000 ? 0 : 1)} Mio.`;
  }
  if (abs >= 1_000) {
    return `${(value / 1_000).toFixed(abs >= 10_000 ? 0 : 1)} Tsd.`;
  }
  return value.toLocaleString("de-DE", { maximumFractionDigits: 2 });
}

/** Backend timestamps come through as e.g. "2007-12-31 00:00:00" — every
 * metric in this app is daily-or-coarser, so the time component is always
 * midnight and never carries information; strip it for axis ticks and the
 * tooltip label so dates aren't twice as long as they need to be. */
function formatDateLabel(value: unknown): string {
  return String(value ?? "").split(/[ T]/)[0];
}

function mergeLayers(layers: ChartLayer[]): Array<Record<string, string | number>> {
  const byDate = new Map<string, Record<string, string | number>>();

  for (const layer of layers) {
    for (const point of layer.data) {
      const row = byDate.get(point.date) ?? { date: point.date };
      row[layer.id] = point.value;
      byDate.set(point.date, row);
    }
  }

  return Array.from(byDate.values()).sort((a, b) =>
    String(a.date).localeCompare(String(b.date))
  );
}

/** Data-driven [min, max] for one axis's layers, compressed/stretched by
 * `zoom` around its center — zoom > 1 narrows the visible range (zoom in),
 * zoom < 1 widens it (zoom out). Mirrors dragging a price axis in
 * TradingView: pulling it apart stretches the scale instead of panning. */
function computeDomain(layers: ChartLayer[], axis: "left" | "right", zoom: number): [number, number] {
  const values = layers.filter((layer) => layer.axis === axis).flatMap((layer) => layer.data.map((p) => p.value));

  if (values.length === 0) return [0, 1];

  const min = Math.min(...values);
  const rawMax = Math.max(...values);
  const max = rawMax === min ? rawMax + 1 : rawMax;
  const center = (min + max) / 2;
  const halfRange = (max - min) / 2 / zoom;

  return [center - halfRange, center + halfRange];
}

type DragState = { axis: "left" | "right"; startY: number; startZoom: number };

/** Flexible multi-series overlay chart — each layer can be an arbitrary
 * metric+symbol combination on its own left/right axis. Different layers
 * rarely share the exact same date set (different symbols, different
 * report dates), so points are merged by date with gaps left for whichever
 * layers have no data on that date (connectNulls bridges those gaps
 * visually instead of breaking the line). Used by ChartLayerBuilder for
 * both single-analysis chart overlays and (later) cross-analysis compare.
 *
 * Each axis has a draggable handle (left edge / right edge) so a user can
 * stretch or compress that axis's scale by hand, TradingView-style, instead
 * of being stuck with whatever range the data happens to span. */
export default function MultiLayerChart({ layers, height = 320 }: Props) {
  const chartTokens = useChartTokens();
  const isMobile = useIsMobile();
  const merged = useMemo(() => mergeLayers(layers), [layers]);
  const hasRightAxis = layers.some((layer) => layer.axis === "right");
  // Schmalere Y-Achsen-Spalte auf Mobile: 64px pro Achse (128px bei zwei
  // Achsen) lässt auf 375px kaum noch Zeichenfläche übrig
  // (RESPONSIVE.md R-P1-6).
  const axisWidth = isMobile ? 42 : 64;

  const [zoom, setZoom] = useState<{ left: number; right: number }>({ left: 1, right: 1 });
  const dragState = useRef<DragState | null>(null);

  const handleDragMove = useCallback((event: MouseEvent) => {
    const drag = dragState.current;
    if (!drag) return;
    // Dragging the handle UP stretches the axis (zoom in); dragging DOWN
    // compresses it (zoom out) — exponential so the gesture feels smooth
    // across both small nudges and large drags.
    const deltaY = drag.startY - event.clientY;
    const factor = Math.min(8, Math.max(0.15, drag.startZoom * Math.exp(deltaY / 150)));
    setZoom((prev) => ({ ...prev, [drag.axis]: factor }));
  }, []);

  const handleDragEnd = useCallback(() => {
    dragState.current = null;
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
    window.removeEventListener("mousemove", handleDragMove);
    window.removeEventListener("mouseup", handleDragEnd);
  }, [handleDragMove]);

  const startDrag = useCallback(
    (axis: "left" | "right") => (event: React.MouseEvent) => {
      // Without this, dragging across the chart's SVG <text> tick labels
      // makes the browser start a native text/image drag-select — that's
      // the stray yellow "forbidden drop" outline the user is hitting, and
      // it fights with our own mousemove-based dragging.
      event.preventDefault();
      document.body.style.userSelect = "none";
      document.body.style.cursor = "ns-resize";
      dragState.current = { axis, startY: event.clientY, startZoom: zoom[axis] };
      window.addEventListener("mousemove", handleDragMove);
      window.addEventListener("mouseup", handleDragEnd);
    },
    [zoom, handleDragMove, handleDragEnd]
  );

  function resetZoom(axis: "left" | "right") {
    setZoom((prev) => ({ ...prev, [axis]: 1 }));
  }

  if (layers.length === 0) {
    return <div style={emptyState}>Noch keine Kennzahl als Layer hinzugefügt.</div>;
  }

  const leftDomain = computeDomain(layers, "left", zoom.left);
  const rightDomain = computeDomain(layers, "right", zoom.right);

  return (
    <div style={{ position: "relative", width: "100%", height }}>
      {/* Zieh-Griffe sind reine Mouse-Event-Handler (mousemove/mouseup) ohne
       * Touch-Äquivalent — auf Touch-Geräten überlagerten sie bisher nur
       * nutzlos den Chart-Rand, ohne je auf einen Tap/Drag zu reagieren
       * (RESPONSIVE.md R-P1-6). Touch-Zoom nachzurüsten wäre ein größerer,
       * eigenständiger Eingriff; bis dahin werden sie auf Mobile ausgeblendet
       * statt eine funktionslose Fläche zu zeigen. */}
      {!isMobile ? (
        <div
          onMouseDown={startDrag("left")}
          onDoubleClick={() => resetZoom("left")}
          onDragStart={(e) => e.preventDefault()}
          title="Ziehen, um die linke Achse zu stauchen/strecken — Doppelklick setzt zurück"
          style={axisHandle("left")}
        />
      ) : null}
      {!isMobile && hasRightAxis ? (
        <div
          onMouseDown={startDrag("right")}
          onDoubleClick={() => resetZoom("right")}
          onDragStart={(e) => e.preventDefault()}
          title="Ziehen, um die rechte Achse zu stauchen/strecken — Doppelklick setzt zurück"
          style={axisHandle("right")}
        />
      ) : null}

      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={merged} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={chartTokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            stroke={chartTokens.axis}
            tick={{ fontSize: isMobile ? 10 : 12 }}
            tickLine={false}
            tickFormatter={formatDateLabel}
            minTickGap={isMobile ? 24 : 8}
            interval="preserveStartEnd"
          />
          <YAxis
            yAxisId="left"
            domain={leftDomain}
            allowDataOverflow
            stroke={chartTokens.axis}
            tick={{ fontSize: isMobile ? 10 : 12 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={formatCompactNumber}
            width={axisWidth}
          />
          {hasRightAxis ? (
            <YAxis
              yAxisId="right"
              orientation="right"
              domain={rightDomain}
              allowDataOverflow
              stroke={chartTokens.axis}
              tick={{ fontSize: isMobile ? 10 : 12 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={formatCompactNumber}
              width={axisWidth}
            />
          ) : null}
          <Tooltip
            contentStyle={{
              background: theme.colors.panel,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.radius.sm,
              color: theme.colors.textPrimary,
            }}
            labelStyle={{ color: theme.colors.textMuted }}
            labelFormatter={formatDateLabel}
            formatter={(value, name) => [
              typeof value === "number" ? formatCompactNumber(value) : String(value ?? ""),
              name,
            ]}
          />
          <Legend wrapperStyle={{ fontSize: "0.8rem" }} />
          {layers.map((layer) => (
            <Line
              key={layer.id}
              yAxisId={layer.axis}
              type="monotone"
              dataKey={layer.id}
              name={layer.label}
              stroke={layer.color}
              strokeWidth={2}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

function axisHandle(axis: "left" | "right"): React.CSSProperties {
  return {
    position: "absolute",
    top: 0,
    bottom: "32px",
    [axis]: 0,
    width: "56px",
    cursor: "ns-resize",
    zIndex: 5,
    userSelect: "none",
    WebkitUserSelect: "none",
    outline: "none",
    WebkitTapHighlightColor: "transparent",
  };
}

const emptyState = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  height: "160px",
  color: theme.colors.textMuted,
  fontSize: "0.92rem",
  border: `1px dashed ${theme.colors.borderSubtle}`,
  borderRadius: theme.radius.md,
};
