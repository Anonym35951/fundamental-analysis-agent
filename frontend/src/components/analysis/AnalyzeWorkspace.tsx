import type { ReactNode } from "react";
import type { AnalysisMode, SymbolMeta } from "../../api/analysis";
import type { CustomAnalysisDefinition } from "../../types/customAnalysis";
import { theme } from "../ui/theme";
import SymbolCommandField from "./SymbolCommandField";
import ModeSelectorGrid from "./ModeSelectorGrid";
import FrequencyToggle from "./FrequencyToggle";
import CustomDefinitionGrid from "../customAnalysis/CustomDefinitionGrid";

type ModeOption = {
  value: AnalysisMode;
  label: string;
  description: string;
};

export type AnalysisTab = "standard" | "individuell";

type Props = {
  symbol: string;
  onSymbolChange: (value: string) => void;
  onSymbolFocus: () => void;
  onSymbolBlur: () => void;
  isSuggestionsOpen: boolean;
  isLoadingSymbols: boolean;
  filteredSuggestions: SymbolMeta[];
  isSymbolSearchDegraded?: boolean;
  onSelectSuggestion: (symbol: string) => void;
  isFavorited: boolean;
  onToggleFavorite: () => void;

  analysisTab: AnalysisTab;
  onSelectAnalysisTab: (tab: AnalysisTab) => void;

  modeOptions: ModeOption[];
  selectedMode: AnalysisMode;
  onSelectMode: (mode: AnalysisMode) => void;

  customDefinitions: CustomAnalysisDefinition[];
  selectedDefinitionId: number | null;
  onSelectCustomDefinition: (definition: CustomAnalysisDefinition) => void;
  onEditCustomDefinition: (definition: CustomAnalysisDefinition) => void;
  onDeleteCustomDefinition: (id: number) => void;
  canCreateNewDefinition: boolean;
  onCreateNewDefinition: () => void;
  onRunAdHoc: () => void;

  selectedFrequency: "annual" | "quarterly";
  onFrequencyChange: (value: "annual" | "quarterly") => void;

  startButton: ReactNode;
};

/** "Research workspace" composition: symbol input dominates as the visual
 * focal point, mode/frequency selection sit below as a subordinate
 * configuration panel — mirrors a terminal's ticker-lookup-first layout
 * rather than a flat multi-field admin form. Standard and Individuell modes
 * share this single workspace (and the same symbol field) instead of living
 * on separate pages. */
export default function AnalyzeWorkspace({
  symbol,
  onSymbolChange,
  onSymbolFocus,
  onSymbolBlur,
  isSuggestionsOpen,
  isLoadingSymbols,
  filteredSuggestions,
  isSymbolSearchDegraded,
  onSelectSuggestion,
  isFavorited,
  onToggleFavorite,
  analysisTab,
  onSelectAnalysisTab,
  modeOptions,
  selectedMode,
  onSelectMode,
  customDefinitions,
  selectedDefinitionId,
  onSelectCustomDefinition,
  onEditCustomDefinition,
  onDeleteCustomDefinition,
  canCreateNewDefinition,
  onCreateNewDefinition,
  onRunAdHoc,
  selectedFrequency,
  onFrequencyChange,
  startButton,
}: Props) {
  return (
    <section style={workspaceCard}>
      <SymbolCommandField
        symbol={symbol}
        onSymbolChange={onSymbolChange}
        onFocus={onSymbolFocus}
        onBlur={onSymbolBlur}
        isSuggestionsOpen={isSuggestionsOpen}
        isLoadingSymbols={isLoadingSymbols}
        filteredSuggestions={filteredSuggestions}
        isSymbolSearchDegraded={isSymbolSearchDegraded}
        onSelectSuggestion={onSelectSuggestion}
        isFavorited={isFavorited}
        onToggleFavorite={onToggleFavorite}
      />

      <div style={section}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "16px", flexWrap: "wrap" }}>
          <div style={eyebrow}>Analysemodus</div>
          <div style={tabSwitcher} data-tour="analyze-mode-tabs">
            <button
              type="button"
              onClick={() => onSelectAnalysisTab("standard")}
              style={tabButton(analysisTab === "standard")}
            >
              Standard
            </button>
            <button
              type="button"
              onClick={() => onSelectAnalysisTab("individuell")}
              style={tabButton(analysisTab === "individuell")}
            >
              Individuell
            </button>
          </div>
        </div>

        <div style={{ marginTop: "14px" }} data-tour="analyze-custom-builder">
          {analysisTab === "standard" ? (
            <ModeSelectorGrid
              modeOptions={modeOptions}
              selectedMode={selectedMode}
              onSelectBuiltIn={onSelectMode}
            />
          ) : (
            <CustomDefinitionGrid
              definitions={customDefinitions}
              selectedDefinitionId={selectedDefinitionId}
              onSelectDefinition={onSelectCustomDefinition}
              onEditDefinition={onEditCustomDefinition}
              onDeleteDefinition={onDeleteCustomDefinition}
              canCreateNew={canCreateNewDefinition}
              onCreateNew={onCreateNewDefinition}
              onRunAdHoc={onRunAdHoc}
            />
          )}
        </div>
      </div>

      <div style={{ ...section, display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "16px" }}>
        <div>
          <div style={eyebrow}>Frequenz</div>
          <div style={{ marginTop: "10px" }}>
            <FrequencyToggle
              value={selectedFrequency}
              onChange={onFrequencyChange}
              disabled={analysisTab === "standard" && selectedMode === "full"}
            />
          </div>
        </div>

        {startButton}
      </div>
    </section>
  );
}

const workspaceCard = {
  background: theme.colors.bgGradientEnd,
  borderRadius: theme.radius.lg,
  padding: "28px",
  border: `1px solid ${theme.colors.borderSubtle}`,
  boxShadow: "0 14px 34px rgba(0, 0, 0, 0.22)",
  display: "flex",
  flexDirection: "column" as const,
  gap: "26px",
};

const section = {
  marginTop: "0px",
};

const eyebrow = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const tabSwitcher: React.CSSProperties = {
  display: "inline-flex",
  gap: "4px",
  padding: "4px",
  borderRadius: theme.radius.pill,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
};

function tabButton(isActive: boolean): React.CSSProperties {
  return {
    padding: "7px 16px",
    borderRadius: theme.radius.pill,
    border: "none",
    fontSize: "0.86rem",
    fontWeight: 700,
    cursor: "pointer",
    background: isActive ? theme.colors.chromeStrong : "transparent",
    color: isActive ? theme.colors.onChrome : theme.colors.textSecondary,
    transition: `background ${theme.motion.fast} ${theme.motion.easing}, color ${theme.motion.fast} ${theme.motion.easing}`,
  };
}
