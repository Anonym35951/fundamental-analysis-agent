import { Check, Pencil, Plus, Trash2, Zap } from "lucide-react";
import { theme } from "../ui/theme";
import Badge from "../ui/Badge";
import type { CustomAnalysisDefinition } from "../../types/customAnalysis";

type Props = {
  definitions: CustomAnalysisDefinition[];
  selectedDefinitionId: number | null;
  onSelectDefinition: (definition: CustomAnalysisDefinition) => void;
  onEditDefinition: (definition: CustomAnalysisDefinition) => void;
  onDeleteDefinition: (id: number) => void;
  canCreateNew: boolean;
  onCreateNew: () => void;
  onRunAdHoc: () => void;
};

/** "Individuell" tab counterpart to ModeSelectorGrid: saved custom
 * definitions as selectable cards, plus a dashed "+" card to build a new
 * one and a dashed "Einmalige Analyse" card for the ad-hoc (no-save) path —
 * both open the same DefinitionBuilder, just in a modal instead of a
 * separate page. */
export default function CustomDefinitionGrid({
  definitions,
  selectedDefinitionId,
  onSelectDefinition,
  onEditDefinition,
  onDeleteDefinition,
  canCreateNew,
  onCreateNew,
  onRunAdHoc,
}: Props) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))",
        gap: "10px",
      }}
    >
      {definitions.map((definition) => {
        const isSelected = definition.id === selectedDefinitionId;
        return (
          <div
            key={definition.id}
            role="button"
            tabIndex={0}
            onClick={() => onSelectDefinition(definition)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") onSelectDefinition(definition);
            }}
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "8px",
              textAlign: "left",
              padding: "16px 16px 14px",
              borderRadius: theme.radius.lg,
              border: `1px solid ${isSelected ? theme.colors.chromeBorder : theme.glass.subtle.border}`,
              background: isSelected ? theme.colors.chromeSoft : theme.glass.subtle.background,
              cursor: "pointer",
              transition: `background ${theme.motion.fast} ${theme.motion.easing}, border-color ${theme.motion.fast} ${theme.motion.easing}`,
            }}
          >
            <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "8px" }}>
              <span style={{ fontSize: "0.96rem", fontWeight: 800, color: theme.colors.textPrimary, lineHeight: 1.3 }}>
                {definition.name}
              </span>
              <div style={{ display: "flex", alignItems: "center", gap: "4px", flexShrink: 0 }}>
                <button
                  type="button"
                  aria-label="Bearbeiten"
                  title="Bearbeiten"
                  onClick={(e) => {
                    e.stopPropagation();
                    onEditDefinition(definition);
                  }}
                  style={iconButtonStyle}
                >
                  <Pencil size={12} color={theme.colors.textMuted} />
                </button>
                <button
                  type="button"
                  aria-label="Löschen"
                  title="Löschen"
                  onClick={(e) => {
                    e.stopPropagation();
                    onDeleteDefinition(definition.id);
                  }}
                  style={iconButtonStyle}
                >
                  <Trash2 size={12} color={theme.colors.textMuted} />
                </button>
                <span
                  style={{
                    width: "20px",
                    height: "20px",
                    borderRadius: theme.radius.pill,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: isSelected ? theme.colors.chromeStrong : "transparent",
                    border: `1px solid ${isSelected ? theme.colors.chromeStrong : theme.colors.border}`,
                  }}
                >
                  {isSelected ? <Check size={12} color={theme.colors.onChrome} strokeWidth={3} /> : null}
                </span>
              </div>
            </div>
            <span style={{ fontSize: "0.84rem", color: theme.colors.textSecondary, lineHeight: 1.5 }}>
              {definition.metric_count} Kennzahlen ausgewählt
            </span>
            <Badge tone="neutral" style={{ alignSelf: "flex-start", marginTop: "2px" }}>
              Eigene Analyse
            </Badge>
          </div>
        );
      })}

      <button
        type="button"
        onClick={onRunAdHoc}
        style={{ ...dashedCardStyle }}
      >
        <Zap size={18} color={theme.colors.chrome} />
        <span style={{ fontSize: "0.92rem", fontWeight: 700, color: theme.colors.textPrimary }}>
          Einmalige Analyse
        </span>
        <span style={{ fontSize: "0.78rem", color: theme.colors.textMuted, lineHeight: 1.4 }}>
          Kennzahlen frei wählen, ohne zu speichern
        </span>
      </button>

      <button
        type="button"
        onClick={onCreateNew}
        disabled={!canCreateNew}
        style={{
          ...dashedCardStyle,
          opacity: canCreateNew ? 1 : 0.5,
          cursor: canCreateNew ? "pointer" : "not-allowed",
        }}
        title={canCreateNew ? undefined : "Free-Plan-Limit erreicht"}
      >
        <Plus size={18} color={theme.colors.chrome} />
        <span style={{ fontSize: "0.92rem", fontWeight: 700, color: theme.colors.textPrimary }}>
          Neue Analyse
        </span>
        <span style={{ fontSize: "0.78rem", color: theme.colors.textMuted, lineHeight: 1.4 }}>
          {canCreateNew ? "Eigene Kennzahlen-Kombination erstellen" : "Free-Plan-Limit erreicht"}
        </span>
      </button>
    </div>
  );
}

const iconButtonStyle: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  border: "none",
  background: "transparent",
  padding: "2px",
  cursor: "pointer",
  opacity: 0.7,
};

const dashedCardStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "flex-start",
  gap: "8px",
  textAlign: "left",
  padding: "16px 16px 14px",
  borderRadius: theme.radius.lg,
  border: `1px dashed ${theme.colors.borderSubtle}`,
  background: "transparent",
};
