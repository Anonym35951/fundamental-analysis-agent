import { theme } from "../ui/theme";

type Props = {
  value: "annual" | "quarterly";
  onChange: (value: "annual" | "quarterly") => void;
  disabled?: boolean;
};

const options: Array<{ value: "annual" | "quarterly"; label: string }> = [
  { value: "annual", label: "Annual" },
  { value: "quarterly", label: "Quarterly" },
];

export default function FrequencyToggle({ value, onChange, disabled = false }: Props) {
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
