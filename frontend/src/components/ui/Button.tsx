import type { ButtonHTMLAttributes, CSSProperties } from "react";
import { theme } from "./theme";

type Variant = "primary" | "secondary" | "ghost" | "danger" | "cta";

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: Variant;
  /** Fully rounded pill shape, matching the new nav/CTA aesthetic. Defaults
   * to on, since most surfaces now use pills — pass false for the rare
   * case a squarer button fits better. */
  pill?: boolean;
};

const baseStyle: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: "8px",
  padding: "11px 22px",
  fontSize: "0.95rem",
  fontWeight: 600,
  cursor: "pointer",
  border: "1px solid transparent",
  transition: `opacity ${theme.motion.fast} ${theme.motion.easing}, transform ${theme.motion.fast} ${theme.motion.easing}, box-shadow ${theme.motion.fast} ${theme.motion.easing}`,
};

const variantStyles: Record<Variant, CSSProperties> = {
  primary: {
    background: theme.colors.panelAlt,
    color: theme.colors.textPrimary,
    borderColor: theme.colors.chromeBorder,
    boxShadow: "0 12px 26px rgba(0, 0, 0, 0.4)",
  },
  secondary: {
    background: theme.colors.panelAlt,
    color: theme.colors.textPrimary,
    borderColor: theme.colors.border,
  },
  ghost: {
    background: "transparent",
    color: theme.colors.textSecondary,
    borderColor: theme.colors.borderSubtle,
  },
  danger: {
    background: theme.colors.dangerSoft,
    color: theme.colors.dangerText,
    borderColor: theme.colors.dangerBorder,
  },
  /** Reserved for the one or two highest-emphasis calls to action per page
   * (e.g. landing hero, plan upgrade) — carries the brushed-chrome gradient.
   * The gradient is light, so text/icon color must stay near-black for
   * contrast (unlike the other dark-filled variants). */
  cta: {
    background: theme.gradients.ctaPrimary,
    color: theme.colors.bgDeep,
    borderColor: theme.colors.chromeBorder,
    boxShadow: "0 14px 30px rgba(0, 0, 0, 0.45)",
  },
};

export default function Button({
  variant = "primary",
  pill = true,
  style,
  disabled,
  ...rest
}: ButtonProps) {
  return (
    <button
      {...rest}
      disabled={disabled}
      style={{
        ...baseStyle,
        borderRadius: pill ? theme.radius.pill : theme.radius.sm,
        ...variantStyles[variant],
        opacity: disabled ? 0.55 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
        ...style,
      }}
    />
  );
}
