import type { AnalysisMode } from "../../api/analysis";
import ModeCard from "./ModeCard";

type ModeOption = {
  value: AnalysisMode;
  label: string;
  description: string;
};

type Props = {
  modeOptions: ModeOption[];
  selectedMode: AnalysisMode;
  onSelectBuiltIn: (mode: AnalysisMode) => void;
};

/** Renders only the built-in (Standard) analysis modes — custom definitions
 * now live in their own "Individuell" tab via CustomDefinitionGrid instead
 * of being mixed into this grid. */
export default function ModeSelectorGrid({ modeOptions, selectedMode, onSelectBuiltIn }: Props) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
        gap: "10px",
      }}
    >
      {modeOptions.map((option) => (
        <ModeCard
          key={option.value}
          label={option.label}
          description={option.description}
          isSelected={selectedMode === option.value}
          onClick={() => onSelectBuiltIn(option.value)}
        />
      ))}
    </div>
  );
}
