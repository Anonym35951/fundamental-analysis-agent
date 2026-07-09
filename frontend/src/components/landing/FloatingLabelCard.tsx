import type { CSSProperties, ReactNode } from "react";
import { motion } from "framer-motion";
import { theme } from "../ui/theme";

type FloatingLabelCardProps = {
  icon: ReactNode;
  title: string;
  description: string;
  /** Optional second line — used to give these cards real feature-card
   * weight (they carry the platform's core value props) instead of reading
   * as small tooltip-like chips. */
  secondaryDescription?: string;
  style?: CSSProperties;
};

/** Feature panel flanking the hero's central mode-switcher — icon, title,
 * and up to two lines of supporting copy. These carry the platform's
 * primary value props (metric engine, custom logic), so they're sized to be
 * equal-or-larger than the central panel, not a small tooltip-like chip.
 * Purely presentational; positioning/floating motion and sizing (grid
 * column width) are the caller's responsibility. */
export default function FloatingLabelCard({
  icon,
  title,
  description,
  secondaryDescription,
  style,
}: FloatingLabelCardProps) {
  return (
    <motion.div
      whileHover={{ y: -4, scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      transition={theme.motion.spring}
      style={{ ...card, ...style }}
    >
      <span style={iconBadge}>{icon}</span>
      <div>
        <div style={titleStyle}>{title}</div>
        <div style={descriptionStyle}>{description}</div>
        {secondaryDescription ? <div style={secondaryStyle}>{secondaryDescription}</div> : null}
      </div>
    </motion.div>
  );
}

const card: CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  gap: "14px",
  padding: "22px 22px",
  borderRadius: theme.radius.lg,
  background: theme.glass.background,
  border: `1px solid ${theme.glass.border}`,
  backdropFilter: `blur(${theme.glass.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.blur})`,
  boxShadow: theme.glass.shadow,
};

const iconBadge: CSSProperties = {
  flexShrink: 0,
  width: "44px",
  height: "44px",
  borderRadius: theme.radius.pill,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: theme.colors.chromeSoft,
  color: theme.colors.chrome,
};

const titleStyle: CSSProperties = {
  fontWeight: 700,
  fontSize: "1.04rem",
  color: theme.colors.textPrimary,
  marginBottom: "8px",
};

const descriptionStyle: CSSProperties = {
  fontSize: "0.88rem",
  lineHeight: 1.6,
  color: theme.colors.textSecondary,
};

const secondaryStyle: CSSProperties = {
  marginTop: "8px",
  fontSize: "0.84rem",
  lineHeight: 1.6,
  color: theme.colors.textMuted,
};
