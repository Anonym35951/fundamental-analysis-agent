export type AnalysisStatus = "running" | "done" | "error";

export type Progress = {
  job_id: string;
  symbol: string;
  status: AnalysisStatus;
  total: number;
  done: number;
  current?: string | null;
  error?: string | null;
};

/** Shape of a single named criterion inside a category payload, e.g. the
 * "net_debt_to_ebitda" entry of a "Optionality|annual" result. */
export type CriterionResult = {
  value: number | string | boolean | null;
  meets_criterion?: boolean;
  message?: string;
  [detail: string]: unknown;
};

/** CRV ("Chance-Risiko-Verhältnis") sub-shape, present on some category
 * payloads (optionality, average-grower, typical-cyclers, ...). */
export type CrvMultipleResult = {
  crv_positive?: boolean;
  crv_conservative?: number;
  crv_aggressive?: number;
  course_targets?: Record<string, number>;
  targets?: Record<string, number>;
  inputs?: Record<string, unknown>;
  error?: string;
};

export type CrvResult = {
  multiples_used?: string[];
  crv_results?: Record<string, CrvMultipleResult>;
};

/** One category's result, keyed as "<DISPLAY_NAME>|<frequency>" in
 * FullResult.results (see DISPLAY_NAME convention in api/routes/analyze.py).
 * Criterion entries are dynamic (one per evaluated metric), captured via the
 * index signature alongside the fixed fields below. */
export type CategoryResult = {
  symbol?: string;
  frequency?: string;
  overall_assessment?: string;
  message?: string;
  error?: string;
  crv?: CrvResult;
  [criterionKey: string]: CriterionResult | CrvResult | string | undefined;
};

export type FullResult = {
  job_id: string;
  symbol: string;
  status: AnalysisStatus;
  total: number;
  done: number;
  error?: string | null;
  results: Record<string, CategoryResult>;
  /** ISO-Berichtswährung der Fundamentaldaten (EVOLVING.md EV-021/022) -
   * `null`/fehlend, wenn unbestimmbar. */
  reporting_currency?: string | null;
};