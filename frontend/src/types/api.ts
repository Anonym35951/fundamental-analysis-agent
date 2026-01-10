//frontend/src/types/api.ts
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

export type FullResult = {
  job_id: string;
  symbol: string;
  status: "running" | "done" | "error";
  total: number;
  done: number;
  error?: string | null;
  results: Record<string, any>;
};