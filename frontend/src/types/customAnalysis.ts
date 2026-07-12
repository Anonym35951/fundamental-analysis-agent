export type CustomAnalysisFrequency = "annual" | "quarterly";

export type MetricParamType = "string" | "number" | "date" | "enum";

export type MetricParamSpec = {
  name: string;
  type: MetricParamType;
  required: boolean;
  default: unknown;
  enum_values: string[] | null;
};

export type MetricResultShape = "scalar" | "dict" | "timeseries" | "complex";

export type MetricCatalogEntry = {
  key: string;
  label: string;
  category: string;
  requires_symbol: boolean;
  result_shape: MetricResultShape;
  params: MetricParamSpec[];
};

export type CriterionOperator = ">" | "<" | ">=" | "<=";

export type MetricCriterion = {
  operator: CriterionOperator;
  threshold: number;
};

export type MetricSelection = {
  key: string;
  params: Record<string, unknown>;
  criterion?: MetricCriterion | null;
};

export type CustomAnalysisStartRequest = {
  symbol: string;
  frequency?: CustomAnalysisFrequency | null;
  metrics: MetricSelection[];
  /** Fürs Event-Logging: unterscheidet Compare-Aufrufe vom Custom-Analysis-
   * Builder. Default (weggelassen) ist "custom_builder". */
  source?: "compare" | "custom_builder";
};

export type CustomAnalysisStartResponse = {
  job_id: string;
  symbol: string;
};

export type CustomMetricResult = {
  value: unknown;
  series?: Array<{ date: string; value: number }>;
  meets_criterion?: boolean;
  error?: string;
};

export type CustomAnalysisResult = {
  job_id: string;
  symbol: string;
  status: "running" | "done" | "error";
  metrics: Record<string, CustomMetricResult>;
};

export type CustomAnalysisDefinition = {
  id: number;
  name: string;
  metrics: MetricSelection[];
  metric_count: number;
  last_run_at: string | null;
  created_at: string;
  updated_at: string;
};

export type CustomAnalysisDefinitionCreate = {
  name: string;
  metrics: MetricSelection[];
};

export type CustomAnalysisDefinitionUpdate = {
  name?: string;
  metrics?: MetricSelection[];
};

export type CustomAnalysisDefinitionRunRequest = {
  symbol: string;
  frequency?: CustomAnalysisFrequency | null;
};
