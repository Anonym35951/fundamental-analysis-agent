import { Check } from "lucide-react";
import { Card, Input, InfoTooltip, Select, theme } from "../ui";
import type { CriterionOperator, MetricCatalogEntry, MetricSelection } from "../../types/customAnalysis";

const OPERATORS: CriterionOperator[] = [">", "<", ">=", "<="];

type Props = {
  entry: MetricCatalogEntry;
  selection?: MetricSelection;
  isSelected: boolean;
  onToggle: (entry: MetricCatalogEntry) => void;
  onParamChange: (key: string, paramName: string, value: string | number | undefined) => void;
  onCriterionChange: (key: string, operator: CriterionOperator | "", threshold: string) => void;
  /** Skips the threshold-criterion select/input block — used by the Compare
   * workspace, which is purely descriptive (no pass/fail evaluation). */
  hideCriterion?: boolean;
};

/** Single selectable metric tile — presentational only, all selection state
 * and param/criterion mutation logic stays in DefinitionBuilder. */
export default function MetricPickerCard({
  entry,
  selection,
  isSelected,
  onToggle,
  onParamChange,
  onCriterionChange,
  hideCriterion = false,
}: Props) {
  return (
    <Card
      variant={isSelected ? "default" : "glass"}
      style={{
        padding: "12px 14px",
        borderColor: isSelected ? theme.colors.chromeBorder : theme.glass.subtle.border,
        background: isSelected ? theme.colors.chromeSoft : theme.glass.subtle.background,
        transition: `background ${theme.motion.fast} ${theme.motion.easing}, border-color ${theme.motion.fast} ${theme.motion.easing}`,
      }}
    >
      <label
        style={{ display: "flex", alignItems: "center", gap: "10px", cursor: "pointer", color: theme.colors.textPrimary }}
        onClick={(e) => {
          e.preventDefault();
          onToggle(entry);
        }}
      >
        <span
          style={{
            flexShrink: 0,
            width: "20px",
            height: "20px",
            borderRadius: theme.radius.pill,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            background: isSelected ? theme.colors.chromeStrong : "transparent",
            border: `1px solid ${isSelected ? theme.colors.chromeStrong : theme.colors.border}`,
            transition: `background ${theme.motion.fast} ${theme.motion.easing}, border-color ${theme.motion.fast} ${theme.motion.easing}`,
          }}
        >
          {isSelected ? <Check size={13} color={theme.colors.onChrome} strokeWidth={3} /> : null}
        </span>
        <span style={{ minWidth: 0, overflowWrap: "break-word" }}>{entry.label}</span>
        <InfoTooltip metricKey={entry.key} />
      </label>

      {isSelected && selection ? (
        <div style={{ marginTop: "10px", display: "flex", flexDirection: "column", gap: "8px" }}>
          {entry.params.map((param) => (
            <div key={param.name} style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
              <span style={{ fontSize: "0.78rem", color: theme.colors.textMuted, minWidth: "90px", flexShrink: 0 }}>{param.name}</span>
              {param.type === "enum" && param.enum_values ? (
                <Select
                  value={String(selection.params[param.name] ?? param.default ?? "")}
                  onChange={(e) => onParamChange(entry.key, param.name, e.target.value)}
                  style={{ flex: "1 1 120px", minWidth: 0 }}
                >
                  {param.enum_values.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </Select>
              ) : (
                <Input
                  type={param.type === "number" ? "number" : param.type === "date" ? "date" : "text"}
                  value={String(selection.params[param.name] ?? "")}
                  onChange={(e) => {
                    if (param.type !== "number") {
                      onParamChange(entry.key, param.name, e.target.value);
                      return;
                    }
                    // Ohne diese Umwandlung landet der rohe String im
                    // params-Objekt (JSON.stringify sendet dann z. B.
                    // "threshold": "75" statt 75) - Model.py-Methoden wie
                    // analyze_payout_ratio vergleichen den Wert direkt
                    // numerisch (`payout_ratio > threshold`) und crashen mit
                    // "'>' not supported between instances of 'float' and
                    // 'str'". Leeres Feld -> undefined statt 0, damit der
                    // Backend-Default greift (mirrort updateCriterion's
                    // Number(threshold) für denselben Zweck beim Kriterium).
                    const raw = e.target.value;
                    onParamChange(entry.key, param.name, raw === "" ? undefined : Number(raw));
                  }}
                  style={{ flex: "1 1 120px", minWidth: 0 }}
                />
              )}
            </div>
          ))}

          {!hideCriterion && (entry.result_shape === "scalar" || entry.result_shape === "dict") ? (
            <div style={{ display: "flex", alignItems: "center", gap: "8px", flexWrap: "wrap" }}>
              <span style={{ fontSize: "0.78rem", color: theme.colors.textMuted, minWidth: "90px", flexShrink: 0 }}>Kriterium</span>
              <Select
                value={selection.criterion?.operator ?? ""}
                onChange={(e) =>
                  onCriterionChange(
                    entry.key,
                    e.target.value as CriterionOperator | "",
                    String(selection.criterion?.threshold ?? "")
                  )
                }
                style={{ flex: "1 1 110px", minWidth: 0 }}
              >
                <option value="">Kein Kriterium</option>
                {OPERATORS.map((op) => (
                  <option key={op} value={op}>
                    {op}
                  </option>
                ))}
              </Select>
              <Input
                type="number"
                placeholder="Schwellenwert"
                value={selection.criterion?.threshold ?? ""}
                onChange={(e) => onCriterionChange(entry.key, selection.criterion?.operator ?? ">", e.target.value)}
                style={{ flex: "1 1 110px", minWidth: 0 }}
              />
            </div>
          ) : null}
        </div>
      ) : null}
    </Card>
  );
}
