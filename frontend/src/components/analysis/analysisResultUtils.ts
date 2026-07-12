/** Backend-Ergebnis-Dict für eine Analysemethode - Top-Level-Metafelder sind
 * bekannt, die restlichen Keys sind dynamisch je Analysemodus (jeweils
 * selbst ein PayloadEntry-artiges Objekt, siehe extractCriteria). */
export type AnalysisPayload = {
  symbol?: string;
  frequency?: string;
  overall_assessment?: string;
  message?: string;
  error?: string;
  crv?: unknown;
  [key: string]: unknown;
};

export type AnalysisItem = {
  key: string;
  name: string;
  frequency: string;
  payload: AnalysisPayload;
};

export type Criterion = {
  key: string;
  value: unknown;
  message?: string;
  meets: boolean | null;
};

/** Shape of a single object-valued payload entry once narrowed - matches
 * the metric-result dicts the backend actually sends (value/message/
 * meets_criterion), not the full range of possible payload shapes. */
type PayloadEntry = {
  value?: unknown;
  message?: unknown;
  meets_criterion?: boolean;
};

/** Pulls every object-valued payload key (excluding the known meta fields)
 * out as a displayable criterion — this is the exact pre-redesign extraction
 * logic, unchanged, so downstream score/summary numbers stay identical. */
export function extractCriteria(payload: AnalysisPayload | undefined): Criterion[] {
  if (!payload || typeof payload !== "object") return [];

  return Object.entries(payload)
    .filter(
      ([key, value]) =>
        !["symbol", "frequency", "overall_assessment", "message", "error", "crv"].includes(key) &&
        value &&
        typeof value === "object"
    )
    .map(([key, rawValue]) => {
      const value = rawValue as PayloadEntry;
      return {
        key,
        value: value?.value,
        message: value?.message ? String(value.message) : undefined,
        meets: value?.meets_criterion === true ? true : value?.meets_criterion === false ? false : null,
      };
    });
}

export function calculateScore(criteria: Array<{ meets: boolean | null }>) {
  const positive = criteria.filter((item) => item.meets === true).length;
  const negative = criteria.filter((item) => item.meets === false).length;
  const neutral = criteria.filter((item) => item.meets === null).length;

  return { total: criteria.length, positive, negative, neutral };
}

export function buildSummary(items: AnalysisItem[]) {
  let positive = 0;
  let negative = 0;
  let neutral = 0;
  let bestAssessment = "";

  for (const item of items) {
    const criteria = extractCriteria(item.payload);
    const score = calculateScore(criteria);

    positive += score.positive;
    negative += score.negative;
    neutral += score.neutral;

    if (!bestAssessment && item.payload?.overall_assessment) {
      bestAssessment = String(item.payload.overall_assessment);
    }
  }

  return { positive, negative, neutral, bestAssessment };
}

/** Display-order grouping only — Kritisch first so failing criteria surface
 * without scrolling, then Erfüllt, then Neutral. Does not alter `meets`
 * values or the score counts computed above. */
export function groupCriteriaByStatus(criteria: Criterion[]) {
  return {
    critical: criteria.filter((item) => item.meets === false),
    positive: criteria.filter((item) => item.meets === true),
    neutral: criteria.filter((item) => item.meets === null),
  };
}
