import { useContext } from "react";
import { CompareContext, type CompareContextValue } from "./compareContextValue";

export function useCompare(): CompareContextValue {
  const context = useContext(CompareContext);
  if (!context) {
    throw new Error("useCompare must be used within a CompareProvider");
  }
  return context;
}
