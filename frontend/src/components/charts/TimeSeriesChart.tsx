import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { theme, useChartTokens } from "../ui/theme";

export type TimeSeriesPoint = { date: string; [seriesKey: string]: number | string };

export type TimeSeriesTarget = {
  label: string;
  value: number;
  color?: string;
};

type TimeSeriesChartProps = {
  data: TimeSeriesPoint[];
  /** Series keys to plot as lines (besides "date"). Defaults to every
   * non-"date" key found on the first data point. */
  seriesKeys?: string[];
  /** Optional horizontal reference lines, e.g. CRV buy/fair/sell targets. */
  targets?: TimeSeriesTarget[];
  height?: number;
  valueFormatter?: (value: number) => string;
};

/** Larger annotated line chart for historical metrics (EV/EBITDA over time,
 * price multiples, etc.) and for rendering named valuation targets (CRV
 * buy/fair/sell cases) as reference lines over a series. */
export default function TimeSeriesChart({
  data,
  seriesKeys,
  targets,
  height = 280,
  valueFormatter,
}: TimeSeriesChartProps) {
  const chartTokens = useChartTokens();

  if (data.length === 0) {
    return null;
  }

  const keys = seriesKeys ?? Object.keys(data[0]).filter((key) => key !== "date");

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid stroke={chartTokens.grid} strokeDasharray="3 3" vertical={false} />
          <XAxis
            dataKey="date"
            stroke={chartTokens.axis}
            tick={{ fontSize: 12 }}
            tickLine={false}
          />
          <YAxis
            stroke={chartTokens.axis}
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={valueFormatter}
          />
          <Tooltip
            contentStyle={{
              background: theme.colors.panel,
              border: `1px solid ${theme.colors.border}`,
              borderRadius: theme.radius.sm,
              color: theme.colors.textPrimary,
            }}
            labelStyle={{ color: theme.colors.textMuted }}
            formatter={(value) => [
              typeof value === "number"
                ? valueFormatter
                  ? valueFormatter(value)
                  : value.toLocaleString()
                : String(value ?? ""),
            ]}
          />
          {keys.length > 1 ? <Legend wrapperStyle={{ fontSize: "0.8rem" }} /> : null}
          {keys.map((key, index) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={chartTokens.series[index % chartTokens.series.length]}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          ))}
          {targets?.map((target) => (
            <ReferenceLine
              key={target.label}
              y={target.value}
              stroke={target.color ?? theme.colors.textMuted}
              strokeDasharray="4 4"
              label={{
                value: target.label,
                position: "insideTopRight",
                fill: target.color ?? theme.colors.textMuted,
                fontSize: 11,
              }}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
