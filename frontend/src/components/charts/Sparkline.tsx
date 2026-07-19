import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { theme, useChartTokens } from "../ui/theme";
import { computeStartEndTicks, formatPriceTick } from "./chartUtils";

export type SparklinePoint = { date: string; value: number };

type SparklineProps = {
  data: SparklinePoint[];
  color?: string;
  height?: number;
  /** EVOLVING.md CH-004: blendet eine minimale Y-Achse mit genau zwei Ticks
   * ein — Wert am Zeitraumstart und aktueller Wert (letzter Punkt). Bewusst
   * keine klassische Achse mit Zwischenwerten, damit die kompakten
   * Dashboard-Karten kompakt bleiben (Produktentscheidung). */
  showStartEndAxis?: boolean;
  /** ISO-Code für die Tick-Beschriftung ("USD" → "$…"), kommt aus der
   * price-history-Response. Nur relevant mit showStartEndAxis. */
  currency?: string | null;
};

/** Compact inline trend line for a single metric. Intentionally minimal
 * (no grid/legend); with `showStartEndAxis` it gains a two-tick price axis
 * for the dashboard favorite cards (EVOLVING.md CH-004). */
export default function Sparkline({ data, color, height = 36, showStartEndAxis, currency }: SparklineProps) {
  const { series } = useChartTokens();

  if (data.length < 2) {
    return null;
  }

  const strokeColor = color ?? series[0];
  const tickFormat = (value: number) => formatPriceTick(value, currency);
  const ticks = showStartEndAxis ? computeStartEndTicks(data, tickFormat) : [];
  const showAxis = ticks.length > 0;

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        {/* margin top/bottom 6 statt 2, sobald die Achse sichtbar ist: die
          * Tick-Labels der Extremwerte würden sonst am oberen/unteren Rand
          * abgeschnitten (CH-004). */}
        <LineChart data={data} margin={{ top: showAxis ? 6 : 2, right: 2, bottom: showAxis ? 6 : 2, left: 2 }}>
          <YAxis
            hide={!showAxis}
            domain={["auto", "auto"]}
            ticks={showAxis ? ticks : undefined}
            tickFormatter={tickFormat}
            width={showAxis ? 46 : undefined}
            tick={{ fontSize: 9 }}
            tickLine={false}
            axisLine={false}
            stroke={theme.colors.textMuted}
          />
          <Tooltip
            contentStyle={{
              background: theme.colors.panel,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.radius.sm,
              fontSize: "0.78rem",
              color: theme.colors.textPrimary,
            }}
            labelStyle={{ color: theme.colors.textMuted }}
            formatter={(value) => [
              typeof value === "number" ? value.toLocaleString() : String(value ?? ""),
              "Wert",
            ]}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke={strokeColor}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
