import { apiRequest } from "./client";
import type { Progress } from "../types/analysis";
import type {
  CustomAnalysisStartRequest,
  CustomAnalysisStartResponse,
  CustomAnalysisResult,
  MetricCatalogEntry,
  CustomAnalysisDefinition,
  CustomAnalysisDefinitionCreate,
  CustomAnalysisDefinitionUpdate,
  CustomAnalysisDefinitionRunRequest,
} from "../types/customAnalysis";

export async function getCustomMetricsCatalog(): Promise<MetricCatalogEntry[]> {
  return apiRequest<MetricCatalogEntry[]>("/analyze/custom/metrics", {
    skipAuth: true,
  });
}

export type MetricHistoryRequest = {
  key: string;
  symbol: string;
  frequency?: "annual" | "quarterly" | null;
  start_date?: string | null;
  end_date?: string | null;
};

export type MetricHistoryResponse = {
  key: string;
  symbol: string;
  value: unknown;
  series?: Array<{ date: string; value: number }>;
  error?: string;
};

/** Synchronous single-metric lookup for an arbitrary symbol — used by the
 * chart-layer builder to overlay metrics/symbols outside the current
 * analysis result, without spawning a full job. */
export async function getMetricHistory(
  request: MetricHistoryRequest
): Promise<MetricHistoryResponse> {
  const params = new URLSearchParams({ key: request.key, symbol: request.symbol });
  if (request.frequency) params.set("frequency", request.frequency);
  if (request.start_date) params.set("start_date", request.start_date);
  if (request.end_date) params.set("end_date", request.end_date);

  return apiRequest<MetricHistoryResponse>(`/analyze/custom/history?${params.toString()}`);
}

export async function startCustomAnalysis(
  request: CustomAnalysisStartRequest
): Promise<CustomAnalysisStartResponse> {
  return apiRequest<CustomAnalysisStartResponse>("/analyze/custom/start", {
    method: "POST",
    body: request,
  });
}

export async function getCustomAnalysisProgress(jobId: string): Promise<Progress> {
  return apiRequest<Progress>(`/analyze/custom/${jobId}/progress`);
}

export async function getCustomAnalysisResult(jobId: string): Promise<CustomAnalysisResult> {
  return apiRequest<CustomAnalysisResult>(`/analyze/custom/${jobId}/result`);
}

export async function listDefinitions(): Promise<CustomAnalysisDefinition[]> {
  return apiRequest<CustomAnalysisDefinition[]>("/analyze/custom/definitions");
}

export async function createDefinition(
  payload: CustomAnalysisDefinitionCreate
): Promise<CustomAnalysisDefinition> {
  return apiRequest<CustomAnalysisDefinition>("/analyze/custom/definitions", {
    method: "POST",
    body: payload,
  });
}

export async function updateDefinition(
  id: number,
  payload: CustomAnalysisDefinitionUpdate
): Promise<CustomAnalysisDefinition> {
  return apiRequest<CustomAnalysisDefinition>(`/analyze/custom/definitions/${id}`, {
    method: "PATCH",
    body: payload,
  });
}

export async function deleteDefinition(id: number): Promise<void> {
  return apiRequest<void>(`/analyze/custom/definitions/${id}`, {
    method: "DELETE",
  });
}

export async function runDefinition(
  id: number,
  payload: CustomAnalysisDefinitionRunRequest
): Promise<CustomAnalysisStartResponse> {
  return apiRequest<CustomAnalysisStartResponse>(`/analyze/custom/definitions/${id}/run`, {
    method: "POST",
    body: payload,
  });
}
