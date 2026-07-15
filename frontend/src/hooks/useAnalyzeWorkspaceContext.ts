import { useContext } from "react";
import { AnalyzeWorkspaceContext, type AnalyzeWorkspaceContextValue } from "./analyzeWorkspaceContextValue";

export function useAnalyzeWorkspace(): AnalyzeWorkspaceContextValue {
  const context = useContext(AnalyzeWorkspaceContext);
  if (!context) {
    throw new Error("useAnalyzeWorkspace must be used within an AnalyzeWorkspaceProvider");
  }
  return context;
}
