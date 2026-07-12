export type ChartLayer = {
  id: string;
  label: string;
  data: Array<{ date: string; value: number }>;
  axis: "left" | "right";
  color: string;
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
