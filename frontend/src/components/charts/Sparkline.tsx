import { Line, LineChart, ResponsiveContainer, Tooltip, YAxis } from "recharts";
import { theme, useChartTokens } from "../ui/theme";

export type SparklinePoint = { date: string; value: number };

type SparklineProps = {
  data: SparklinePoint[];
  color?: string;
  height?: number;
};

/** Compact inline trend line for a single metric, used inside MetricResultCard
 * when a metric has an associated historical series. Intentionally minimal
 * (no axes/grid) — for a fully annotated chart use TimeSeriesChart instead. */
export default function Sparkline({ data, color, height = 36 }: SparklineProps) {
  const { series } = useChartTokens();

  if (data.length < 2) {
    return null;
  }

  const strokeColor = color ?? series[0];

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <YAxis hide domain={["auto", "auto"]} />
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
