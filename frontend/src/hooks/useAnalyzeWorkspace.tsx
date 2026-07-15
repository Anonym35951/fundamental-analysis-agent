import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import type { AnalysisTab } from "../components/analysis/AnalyzeWorkspace";
import {
  AnalyzeWorkspaceContext,
  type AnalyzeWorkspaceContextValue,
  type LastResultKind,
} from "./analyzeWorkspaceContextValue";

/** Hält nur die kleinen Pointer/Flags (Tab, Job-IDs, zuletzt gezeigtes
 * Ergebnis) fuer die Analyze-Seite - die eigentlichen Job-Ergebnisse liegen
 * bereits im AnalysisJobsProvider (ebenfalls oberhalb des Routers). Analog zu
 * CompareProvider gemountet, damit AnalyzePage beim Wegnavigieren (Account,
 * Support, Dashboard, ...) nicht per Route-Unmount seinen State verliert
 * (KORREKTUREN.md Punkt 2). Kein localStorage noetig, da die Werte klein sind
 * und nur die laufende Session ueberdauern muessen. */
export function AnalyzeWorkspaceProvider({ children }: { children: ReactNode }) {
  const [analysisTab, setAnalysisTab] = useState<AnalysisTab>("standard");
  const [currentStandardJobId, setCurrentStandardJobId] = useState<string | null>(null);
  const [currentCustomJobId, setCurrentCustomJobId] = useState<string | null>(null);
  const [lastResultKind, setLastResultKind] = useState<LastResultKind>(null);

  const clearAll = useCallback(() => {
    setAnalysisTab("standard");
    setCurrentStandardJobId(null);
    setCurrentCustomJobId(null);
    setLastResultKind(null);
  }, []);

  // Gleiche Logout-Konvention wie CompareProvider: api/client.ts dispatcht
  // dieses Event von jedem Logout-Pfad aus (Button, Inactivity-Timeout,
  // globaler 401-Handler), ohne selbst React-Context-Zugriff zu haben.
  useEffect(() => {
    window.addEventListener("app:logout", clearAll);
    return () => window.removeEventListener("app:logout", clearAll);
  }, [clearAll]);

  const value = useMemo<AnalyzeWorkspaceContextValue>(
    () => ({
      analysisTab,
      setAnalysisTab,
      currentStandardJobId,
      setCurrentStandardJobId,
      currentCustomJobId,
      setCurrentCustomJobId,
      lastResultKind,
      setLastResultKind,
    }),
    [analysisTab, currentStandardJobId, currentCustomJobId, lastResultKind]
  );

  return <AnalyzeWorkspaceContext.Provider value={value}>{children}</AnalyzeWorkspaceContext.Provider>;
}
