import type { HTMLAttributes } from "react";
import { theme } from "./theme";

type Tone = "success" | "danger" | "neutral" | "accent";

type BadgeProps = HTMLAttributes<HTMLSpanElement> & {
  tone?: Tone;
};

const toneStyles: Record<Tone, { background: string; color: string; border: string }> = {
  success: {
    background: theme.colors.successSoft,
    color: theme.colors.successText,
    border: theme.colors.successBorder,
  },
  danger: {
    background: theme.colors.dangerSoft,
    color: theme.colors.dangerText,
    border: theme.colors.dangerBorder,
  },
  neutral: {
    background: theme.colors.borderSubtle,
    color: theme.colors.textSecondary,
    border: theme.colors.border,
  },
  /** Kept as "accent" prop name to avoid touching every call site — now
   * resolves to the neutral chrome tokens internally instead of blue. */
  accent: {
    background: theme.colors.chromeSoft,
    color: theme.colors.textPrimary,
    border: theme.colors.chromeBorder,
  },
};

export default function Badge({ tone = "neutral", style, ...rest }: BadgeProps) {
  const toneStyle = toneStyles[tone];

  return (
    <span
      {...rest}
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "4px 12px",
        borderRadius: theme.radius.pill,
        fontSize: "0.8rem",
        fontWeight: 600,
        background: toneStyle.background,
        color: toneStyle.color,
        border: `1px solid ${toneStyle.border}`,
        ...style,
      }}
    />
  );
}
