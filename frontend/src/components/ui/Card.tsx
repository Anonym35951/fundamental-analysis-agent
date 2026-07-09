import type { CSSProperties, HTMLAttributes } from "react";
import { theme } from "./theme";

type CardProps = HTMLAttributes<HTMLDivElement> & {
  /** "alt" = slightly lighter panel tone for nested/secondary cards.
   * "glass" = frosted glass surface for content over the ambient background.
   * "elevated" = heaviest glass tier, reserved for hero/modal-level cards. */
  variant?: "default" | "alt" | "glass" | "elevated";
};

export default function Card({ variant = "default", style, ...rest }: CardProps) {
  const variantStyle: CSSProperties =
    variant === "glass"
      ? {
          background: theme.glass.subtle.background,
          border: `1px solid ${theme.glass.subtle.border}`,
          backdropFilter: `blur(${theme.glass.subtle.blur})`,
          WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
        }
      : variant === "elevated"
        ? {
            background: theme.glass.elevated.background,
            border: `1px solid ${theme.glass.elevated.border}`,
            backdropFilter: `blur(${theme.glass.elevated.blur})`,
            WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
            boxShadow: theme.glass.elevated.shadow,
          }
        : {
            background: variant === "alt" ? theme.colors.panelAlt : theme.colors.panel,
            border: `1px solid ${theme.colors.border}`,
          };

  return (
    <div
      {...rest}
      style={{
        borderRadius: theme.radius.lg,
        padding: "24px",
        color: theme.colors.textPrimary,
        ...variantStyle,
        ...style,
      }}
    />
  );
}
