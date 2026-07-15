import { createContext } from "react";
import type { AnalysisTab } from "../components/analysis/AnalyzeWorkspace";

/** Welches der beiden Job-Slots (Standard/Individuell) zuletzt eine Analyse
 * gestartet hat - bestimmt, welches Ergebnis angezeigt wird, unabhängig
 * davon, welcher Analysemodus-Tab gerade aktiv ist (KORREKTUREN.md Punkt 2). */
export type LastResultKind = "standard" | "individuell" | null;

export type AnalyzeWorkspaceContextValue = {
  analysisTab: AnalysisTab;
  setAnalysisTab: (tab: AnalysisTab) => void;
  currentStandardJobId: string | null;
  setCurrentStandardJobId: (id: string | null) => void;
  currentCustomJobId: string | null;
  setCurrentCustomJobId: (id: string | null) => void;
  lastResultKind: LastResultKind;
  setLastResultKind: (kind: LastResultKind) => void;
};

// Eigene Datei ohne Komponenten-Export (react-refresh/only-export-components,
// LAUNCH_AUDIT.md P2-10) - spiegelt analysisJobsContextValue.ts/compareContextValue.ts.
export const AnalyzeWorkspaceContext = createContext<AnalyzeWorkspaceContextValue | null>(null);
