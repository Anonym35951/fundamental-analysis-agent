import { theme } from "../ui/theme";
import type { Frequency } from "../../types/frequency";

type Props = {
  value: Frequency;
  onChange: (value: Frequency) => void;
  disabled?: boolean;
  /** EV-134: welche Optionen angezeigt werden - Default alle drei. Kontexte,
   * in denen "ttm" fachlich nicht sinnvoll ist (z. B. eine einzelne
   * Dividenden-Metrik, siehe agent/frequency.py::TTM_CAPABLE_METRICS),
   * können hier gezielt einschränken, statt eine eigene Toggle-Variante zu
   * bauen. */
  availableFrequencies?: readonly Frequency[];
};

const ALL_OPTIONS: Array<{ value: Frequency; label: string }> = [
  { value: "annual", label: "Annual" },
  { value: "ttm", label: "TTM" },
  { value: "quarterly", label: "Quarterly" },
];

export default function FrequencyToggle({
  value,
  onChange,
  disabled = false,
  availableFrequencies = ["annual", "ttm", "quarterly"],
}: Props) {
  const options = ALL_OPTIONS.filter((option) => availableFrequencies.includes(option.value));

  return (
    <div
      style={{
        display: "inline-flex",
        padding: "4px",
        borderRadius: theme.radius.pill,
        background: theme.glass.subtle.background,
        border: `1px solid ${theme.glass.subtle.border}`,
        opacity: disabled ? 0.55 : 1,
      }}
    >
      {options.map((option) => {
        const isActive = value === option.value;

        return (
          <button
            key={option.value}
            type="button"
            disabled={disabled}
            onClick={() => onChange(option.value)}
            title={option.value === "ttm" ? "TTM (letzte 12 Monate): Summe der letzten vier berichteten Quartale" : undefined}
            style={{
              padding: "8px 18px",
              borderRadius: theme.radius.pill,
              border: "none",
              fontSize: "0.88rem",
              fontWeight: 700,
              cursor: disabled ? "not-allowed" : "pointer",
              background: isActive ? theme.colors.chromeStrong : "transparent",
              color: isActive ? theme.colors.onChrome : theme.colors.textSecondary,
              transition: `background ${theme.motion.fast} ${theme.motion.easing}, color ${theme.motion.fast} ${theme.motion.easing}`,
            }}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
