import type { CSSProperties } from "react";
import { theme } from "../ui/theme";

type StatReadoutProps = {
  label: string;
  value: string;
};

/** Small label/value micro-stat, used in the hero's bottom-right strip.
 * Every usage must be a real, static product fact — never framed as
 * live/sensor data. */
export default function StatReadout({ label, value }: StatReadoutProps) {
  return (
    <div style={wrapper}>
      <div style={labelStyle}>{label}</div>
      <div style={valueStyle}>{value}</div>
    </div>
  );
}

const wrapper: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "2px",
};

const labelStyle: CSSProperties = {
  fontSize: "0.7rem",
  fontWeight: 700,
  letterSpacing: "0.04em",
  textTransform: "uppercase",
  color: theme.colors.textMuted,
};

const valueStyle: CSSProperties = {
  fontSize: "0.92rem",
  fontWeight: 700,
  color: theme.colors.textPrimary,
};
