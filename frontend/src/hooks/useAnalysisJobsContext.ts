import { useContext } from "react";
import { AnalysisJobsContext, type AnalysisJobsContextValue } from "./analysisJobsContextValue";

export function useAnalysisJobs(): AnalysisJobsContextValue {
  const context = useContext(AnalysisJobsContext);
  if (!context) {
    throw new Error("useAnalysisJobs must be used within an AnalysisJobsProvider");
  }
  return context;
}
