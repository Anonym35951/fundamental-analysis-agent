import { useCallback, useMemo, useRef, useState } from "react";
import {
  Bar,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { theme, useChartTokens } from "../ui/theme";
import { useIsMobile } from "../../hooks/useMediaQuery";
import { formatCompactNumber, layersCurrencyState, mergeLayers, type BucketMode, type ChartLayer, type ChartType } from "./chartUtils";
import ChartTooltip, { ChartTooltipCard } from "./ChartTooltip";

export type { ChartLayer } from "./chartUtils";

type Props = {
  layers: ChartLayer[];
  height?: number;
  /** Wie die X-Achse Zeitpunkte über mehrere Firmen-Layer hinweg
   * zusammenfasst (EVOLVING.md EV-030) - Default "date" (kein Bucketing,
   * bisheriges Verhalten) ist für Einzelfirmen-Charts korrekt; Compare-
   * Charts geben je nach Annual/Quarterly-Auswahl "year"/"quarter" durch. */
  bucketMode?: BucketMode;
  /** EVOLVING.md CHART-003: reine Render-Konfiguration - "line" (Default,
   * unverändertes bisheriges Verhalten) oder "bar". Ändert keine Daten,
   * keine Berechnung, keinen Zeitraum, keine Währung - beide Modi lesen
   * dieselbe `merged`-Row über denselben `dataKey`. Aufrufer dürfen "bar"
   * nur setzen, wenn `supportedChartTypes` (chartUtils.ts) es erlaubt. */
  chartType?: ChartType;
};

/** Data-driven [min, max] for one axis's layers, compressed/stretched by
 * `zoom` around its center — zoom > 1 narrows the visible range (zoom in),
 * zoom < 1 widens it (zoom out). Mirrors dragging a price axis in
 * TradingView: pulling it apart stretches the scale instead of panning.
 * `includeZero` (EVOLVING.md CHART-003) forces 0 into the domain — required
 * for the bar chart's baseline (unstretched bars must start at 0, otherwise
 * their height would misrepresent the value, and negative values like a
 * loss-making year need a visible zero line to render below). Line mode
 * never passes `includeZero`, so its domain math is unchanged. */
function computeDomain(
  layers: ChartLayer[],
  axis: "left" | "right",
  zoom: number,
  includeZero = false
): [number, number] {
  const values = layers.filter((layer) => layer.axis === axis).flatMap((layer) => layer.data.map((p) => p.value));

  if (values.length === 0) return [0, 1];

  let min = Math.min(...values);
  let rawMax = Math.max(...values);
  if (includeZero) {
    min = Math.min(min, 0);
    rawMax = Math.max(rawMax, 0);
  }
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
export default function MultiLayerChart({ layers, height = 320, bucketMode = "date", chartType = "line" }: Props) {
  const chartTokens = useChartTokens();
  const isMobile = useIsMobile();
  const merged = useMemo(() => mergeLayers(layers, bucketMode), [layers, bucketMode]);
  const hasRightAxis = layers.some((layer) => layer.axis === "right");
  const isBar = chartType === "bar";
  // EVOLVING.md CHART-003: dieselbe Konvention wie StackedCards.tsx/
  // AnimatedNumber.tsx - Säulen-Animation nur, wenn der Nutzer keine
  // reduzierte Bewegung eingestellt hat.
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  }, []);
  // EVOLVING.md CHART-007: je mehr gruppierte Säulen pro Bucket (bis zu 4
  // Firmen, CHART-006) und je schmaler der Viewport, desto enger müssen die
  // einzelnen Balken sein, damit z. B. Quarterly-Daten mit 4 Firmen nicht
  // horizontal überlaufen. Zwei-stufige Verschmälerung statt einer
  // kontinuierlichen Formel - einfach nachvollziehbar.
  const barMaxSize = isMobile ? (layers.length > 2 ? 14 : 24) : layers.length > 2 ? 28 : 40;
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

  // EVOLVING.md CH-002: Ohne Interaktion zeigt ein Overlay die Tooltip-Karte
  // mit dem NEUESTEN Datenpunkt (Werte sichtbar ohne Hover — v.a. mobil, wo
  // man bisher präzise tippen musste). recharts' eigenes defaultIndex kehrt
  // nach mouseleave nie zum Default zurück (combineTooltipInteractionState:
  // defaultIndex-Zweig läuft nur vor der ersten Interaktion) und touchend
  // dispatcht gar nichts (klebender Tooltip) — daher dieser kontrollierte
  // Zustand statt defaultIndex/active-Props.
  const [isInteracting, setIsInteracting] = useState(false);

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

  // EVOLVING.md CHART-003: im Bar-Modus wird die Nulllinie erzwungen
  // (includeZero) und der Achsen-Drag-Zoom deaktiviert - ein verschobener
  // Nullpunkt würde Säulenhöhen visuell verfälschen. Zoom fest auf 1, die
  // per Drag im Linien-Modus gesetzten Zoom-Werte bleiben im State erhalten
  // (rein optische Präferenz, keine Daten) und wirken wieder, sobald zurück
  // auf "line" gewechselt wird.
  const leftDomain = computeDomain(layers, "left", isBar ? 1 : zoom.left, isBar);
  const rightDomain = computeDomain(layers, "right", isBar ? 1 : zoom.right, isBar);

  return (
    <div style={{ position: "relative", width: "100%" }}>
      {"uniform" in currencyState ? (
        <div style={currencySubtitleStyle}>Werte in {currencyState.uniform}</div>
      ) : "mixed" in currencyState ? (
        <div style={currencyMixedBadgeStyle}>
          Originalwährungen: {currencyState.mixed.join(", ")} – Werte nicht direkt vergleichbar
        </div>
      ) : null}
      <div
        style={{ position: "relative", width: "100%", height }}
        // Touch bewusst am Wrapper statt am LineChart: recharts dispatcht bei
        // touchend nichts (der Hover-Zustand bliebe kleben) — touchend hier
        // stellt das Default-Overlay auf jedem Gerät wieder her (CH-002).
        onTouchStart={() => setIsInteracting(true)}
        onTouchEnd={() => setIsInteracting(false)}
        onTouchCancel={() => setIsInteracting(false)}
      >
      {/* Zieh-Griffe sind reine Mouse-Event-Handler (mousemove/mouseup) ohne
       * Touch-Äquivalent — auf Touch-Geräten überlagerten sie bisher nur
       * nutzlos den Chart-Rand, ohne je auf einen Tap/Drag zu reagieren
       * (RESPONSIVE.md R-P1-6). Touch-Zoom nachzurüsten wäre ein größerer,
       * eigenständiger Eingriff; bis dahin werden sie auf Mobile ausgeblendet
       * statt eine funktionslose Fläche zu zeigen. Im Bar-Modus (CHART-003)
       * ebenfalls ausgeblendet - die Nulllinie ist dort fest, kein Zoom. */}
      {!isMobile && !isBar ? (
        <div
          onMouseDown={startDrag("left")}
          onDoubleClick={() => resetZoom("left")}
          onDragStart={(e) => e.preventDefault()}
          title="Ziehen, um die linke Achse zu stauchen/strecken — Doppelklick setzt zurück"
          style={axisHandle("left")}
        />
      ) : null}
      {!isMobile && !isBar && hasRightAxis ? (
        <div
          onMouseDown={startDrag("right")}
          onDoubleClick={() => resetZoom("right")}
          onDragStart={(e) => e.preventDefault()}
          title="Ziehen, um die rechte Achse zu stauchen/strecken — Doppelklick setzt zurück"
          style={axisHandle("right")}
        />
      ) : null}

      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart
          data={merged}
          margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
          onMouseMove={() => setIsInteracting(true)}
          onMouseLeave={() => setIsInteracting(false)}
          // EVOLVING.md CHART-007: schmalerer Abstand zwischen Buckets auf
          // Mobile lässt mehr Platz für die Balken selbst (barMaxSize) statt
          // für Leerraum - wirkt nur im Bar-Modus sichtbar (Line hat keine
          // Kategorie-Balken), ist aber unabhängig von chartType gesetzt, da
          // Recharts die Prop bei LineChart-Serien ignoriert.
          barCategoryGap={isMobile ? "12%" : "20%"}
        >
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
            // active={false} unterdrückt den recharts-Tooltip im Ruhezustand
            // vollständig (auch den nach Touch festklebenden) — das Overlay
            // unten übernimmt dann; undefined = normales Hover-Verhalten.
            active={isInteracting ? undefined : false}
            content={(props) => (
              <ChartTooltip {...props} layers={layers} showCurrencyPerRow={"mixed" in currencyState} />
            )}
          />
          <Legend wrapperStyle={{ fontSize: "0.8rem" }} />
          {layers.map((layer) =>
            isBar ? (
              // EVOLVING.md CHART-003: liest denselben `merged`-Row über
              // denselben dataKey/yAxisId wie die Line unten - reine
              // Render-Alternative desselben Datensatzes, keine zweite
              // Transformation. maxBarSize hält Säulen bei wenigen
              // Datenpunkten (z. B. 3 Jahre) schlank statt Chart-breit.
              <Bar
                key={layer.id}
                yAxisId={layer.axis}
                dataKey={layer.id}
                name={layer.label}
                fill={layer.color}
                radius={[3, 3, 0, 0]}
                maxBarSize={barMaxSize}
                // EVOLVING.md CHART-007: auf Mobile keine Animation, auch
                // ohne prefers-reduced-motion - viele gruppierte Balken
                // gleichzeitig einzublenden ist auf schwächeren Geräten
                // spürbar ruckelig.
                isAnimationActive={!prefersReducedMotion && !isMobile}
              />
            ) : (
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
            )
          )}
        </ComposedChart>
      </ResponsiveContainer>

      {/* EVOLVING.md CH-002: Default-Anzeige des NEUESTEN Datenpunkts (Datum
        * + Werte aller Serien) in identischer Tooltip-Optik, solange nicht
        * gehovert/getippt wird. Rechts verankert (entspricht recharts'
        * eigener defaultIndex-Platzierung am letzten Punkt) — kann dadurch
        * nie über den rechten Rand hinauslaufen. pointerEvents:none hält
        * Chart und Achsen-Drag-Handles (zIndex 5) voll bedienbar. */}
      {!isInteracting && merged.length > 0 ? (
        <div
          style={{
            position: "absolute",
            right: 16 + (hasRightAxis ? axisWidth : 0),
            top: "50%",
            transform: "translateY(-50%)",
            pointerEvents: "none",
            zIndex: 4,
            maxHeight: "90%",
            overflow: "hidden",
          }}
        >
          <ChartTooltipCard
            row={merged[merged.length - 1]}
            layers={layers}
            showCurrencyPerRow={"mixed" in currencyState}
          />
        </div>
      ) : null}
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
