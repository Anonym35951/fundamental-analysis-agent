import { Check } from "lucide-react";
import { theme } from "../ui/theme";

type Props = {
  label: string;
  description: string;
  isSelected: boolean;
  onClick: () => void;
};

export default function ModeCard({ label, description, isSelected, onClick }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
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
        <span
          style={{
            fontSize: "0.96rem",
            fontWeight: 800,
            color: theme.colors.textPrimary,
            lineHeight: 1.3,
          }}
        >
          {label}
        </span>

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
          }}
        >
          {isSelected ? <Check size={12} color={theme.colors.onChrome} strokeWidth={3} /> : null}
        </span>
      </div>

      <span
        style={{
          fontSize: "0.84rem",
          color: theme.colors.textSecondary,
          lineHeight: 1.5,
        }}
      >
        {description}
      </span>
    </button>
  );
}
