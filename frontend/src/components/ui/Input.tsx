import type { InputHTMLAttributes } from "react";
import { theme } from "./theme";

type InputProps = InputHTMLAttributes<HTMLInputElement>;

export default function Input({ style, ...rest }: InputProps) {
  return (
    <input
      {...rest}
      onFocus={(e) => {
        e.currentTarget.style.borderColor = theme.colors.chromeBorder;
        e.currentTarget.style.boxShadow = `0 0 0 3px ${theme.colors.chromeSoft}`;
        rest.onFocus?.(e);
      }}
      onBlur={(e) => {
        e.currentTarget.style.borderColor = theme.glass.subtle.border;
        e.currentTarget.style.boxShadow = "none";
        rest.onBlur?.(e);
      }}
      style={{
        width: "100%",
        padding: "11px 16px",
        borderRadius: theme.radius.md,
        border: `1px solid ${theme.glass.subtle.border}`,
        background: theme.glass.subtle.background,
        color: theme.colors.textPrimary,
        fontSize: "0.95rem",
        outline: "none",
        boxSizing: "border-box",
        transition: `border-color ${theme.motion.fast} ${theme.motion.easing}, box-shadow ${theme.motion.fast} ${theme.motion.easing}`,
        ...style,
      }}
    />
  );
}
