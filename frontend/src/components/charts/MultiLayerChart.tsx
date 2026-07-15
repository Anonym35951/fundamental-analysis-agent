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
import { formatCompactNumber, layersCurrencyState, mergeLayers, type BucketMode, type ChartLayer } from "./chartUtils";
import ChartTooltip from "./ChartTooltip";

export type { ChartLayer } from "./chartUtils";

type Props = {
  layers: ChartLayer[];
  height?: number;
  /** Wie die X-Achse Zeitpunkte über mehrere Firmen-Layer hinweg
   * zusammenfasst (EVOLVING.md EV-030) - Default "date" (kein Bucketing,
   * bisheriges Verhalten) ist für Einzelfirmen-Charts korrekt; Compare-
   * Charts geben je nach Annual/Quarterly-Auswahl "year"/"quarter" durch. */
  bucketMode?: BucketMode;
};

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
export default function MultiLayerChart({ layers, height = 320, bucketMode = "date" }: Props) {
  const chartTokens = useChartTokens();
  const isMobile = useIsMobile();
  const merged = useMemo(() => mergeLayers(layers, bucketMode), [layers, bucketMode]);
  const hasRightAxis = layers.some((layer) => layer.axis === "right");
  // EVOLVING.md EV-023: einheitliche Währung -> dezente Unterzeile (einzige
  // sichtbare Änderung im Standardfall); gemischte Originalwährungen ->
  // Hinweis-Badge + Currency-Code je Tooltip-Zeile; keine bekannte Währung
  // (z. B. Ratio-/Margen-Charts) -> unverändertes bisheriges Verhalten.
  const currencyState = useMemo(() => layersCurrencyState(layers), [layers]);
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
      // { once: true } statt manuellem removeEventListener(handleDragEnd) in
      // handleDragEnd selbst - eine Funktion, die sich per Namen selbst aus
      // einem Listener entfernt, war ein "used before declared"-ESLint-Fund
      // (react-hooks/immutability); mouseup feuert ohnehin nur einmal pro
      // Drag-Geste, once:true entfernt den Listener automatisch nach dem
      // ersten Auftreten.
      window.addEventListener("mouseup", handleDragEnd, { once: true });
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
    <div style={{ position: "relative", width: "100%" }}>
      {"uniform" in currencyState ? (
        <div style={currencySubtitleStyle}>Werte in {currencyState.uniform}</div>
      ) : "mixed" in currencyState ? (
        <div style={currencyMixedBadgeStyle}>
          Originalwährungen: {currencyState.mixed.join(", ")} – Werte nicht direkt vergleichbar
        </div>
      ) : null}
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
            dataKey="label"
            stroke={chartTokens.axis}
            tick={{ fontSize: isMobile ? 10 : 12 }}
            tickLine={false}
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
          {/* EV-031: eigener Tooltip statt Recharts-Standard - der listet nur
             Serien mit definiertem Wert an der gehoverten Position (Root
             Cause des "nur 1-2 Firmen im Tooltip"-Bugs); ChartTooltip
             iteriert stattdessen über `layers` und zeigt "–" für Lücken. */}
          <Tooltip
            content={(props) => (
              <ChartTooltip {...props} layers={layers} showCurrencyPerRow={"mixed" in currencyState} />
            )}
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
    </div>
  );
}

const currencySubtitleStyle: React.CSSProperties = {
  marginBottom: "6px",
  color: theme.colors.textMuted,
  fontSize: "0.78rem",
};

const currencyMixedBadgeStyle: React.CSSProperties = {
  marginBottom: "8px",
  padding: "6px 10px",
  borderRadius: theme.radius.sm,
  background: theme.colors.dangerSoft,
  border: `1px solid ${theme.colors.dangerBorder}`,
  color: theme.colors.dangerText,
  fontSize: "0.8rem",
  lineHeight: 1.5,
};

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
