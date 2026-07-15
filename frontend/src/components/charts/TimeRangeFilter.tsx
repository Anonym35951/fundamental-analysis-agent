import { theme } from "../ui/theme";
import { TIME_RANGE_OPTIONS, type TimeRange } from "./chartUtils";

type Props = {
  value: TimeRange;
  onChange: (value: TimeRange) => void;
  /** Konfigurierbare Teilmenge der 8 Ranges (EVOLVING.md EV-040) - z. B.
   * Fundamental-Charts blenden 1M-6M aus, weil sie bei jährlichen/
   * quartalsweisen Daten fachlich sinnlos wären (EV-041). */
  options?: TimeRange[];
  disabled?: boolean;
};

/** Buttonleiste zur Zeitraumwahl, optisch an `FrequencyToggle` angelehnt.
 * Horizontal scrollbar statt umbrechend, damit auch alle 8 Optionen auf
 * schmalen Mobile-Viewports nutzbar bleiben (RESPONSIVE.md-Muster). */
export default function TimeRangeFilter({ value, onChange, options, disabled = false }: Props) {
  const visibleOptions = options
    ? TIME_RANGE_OPTIONS.filter((option) => options.includes(option.value))
    : TIME_RANGE_OPTIONS;

  return (
    <div
      style={{
        display: "flex",
        overflowX: "auto",
        WebkitOverflowScrolling: "touch",
        scrollbarWidth: "none",
        padding: "4px",
        borderRadius: theme.radius.pill,
        background: theme.glass.subtle.background,
        border: `1px solid ${theme.glass.subtle.border}`,
        opacity: disabled ? 0.55 : 1,
      }}
    >
      {visibleOptions.map((option) => {
        const isActive = value === option.value;

        return (
          <button
            key={option.value}
            type="button"
            disabled={disabled}
            onClick={() => onChange(option.value)}
            style={{
              flex: "0 0 auto",
              padding: "8px 14px",
              borderRadius: theme.radius.pill,
              border: "none",
              fontSize: "0.86rem",
              fontWeight: 700,
              whiteSpace: "nowrap",
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
