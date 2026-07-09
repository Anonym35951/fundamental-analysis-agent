/** One metric-for-one-company line in the comparison. Every company in the
 * workspace shares the exact same metric selection, so `metricKey` is the
 * raw catalog key (no namespacing needed) — that's what makes the pivot
 * table's rows line up cleanly across companies. */
export type CompareLayer = {
  id: string;
  groupId: string;     // = symbol
  groupLabel: string;  // = symbol
  metricKey: string;
  label: string;
  axis: "left" | "right";
  color: string;
  /** True only when `data` has at least one point. */
  chartEligible: boolean;
  data?: Array<{ date: string; value: number }>;
  value: unknown;
  error?: string | null;
};

/** Convenience type for rendering one column per company without
 * re-deriving the metadata from the first layer in the group every time.
 * Column order follows the `companies` array order (insertion order) — no
 * separate timestamp needed. */
export type CompareGroupMeta = {
  groupId: string;     // = symbol
  groupLabel: string;  // = symbol
};
