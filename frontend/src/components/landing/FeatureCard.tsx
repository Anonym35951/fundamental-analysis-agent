import type { CSSProperties, ReactNode } from "react";
import { motion } from "framer-motion";
import { theme } from "../ui/theme";

const sansFont = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

type FeatureCardProps = {
  icon: ReactNode;
  title: string;
  text: string;
};

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0 },
};

/** Value-section card — icon + title + text, with a thin top accent line
 * for the "tech spec" look. Replaces the ad-hoc inline-styled card that
 * used to live directly in `LandingPage.tsx`. */
export default function FeatureCard({ icon, title, text }: FeatureCardProps) {
  return (
    <motion.div
      variants={fadeUp}
      transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1], y: theme.motion.spring, scale: theme.motion.spring }}
      whileHover={{ y: -4, scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      style={card}
    >
      <div style={topAccent} />
      <span style={iconBadge}>{icon}</span>
      <h3 style={titleStyle}>{title}</h3>
      <p style={textStyle}>{text}</p>
    </motion.div>
  );
}

const card: CSSProperties = {
  position: "relative",
  background: theme.glass.subtle.background,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
  borderRadius: theme.radius.lg,
  border: `1px solid ${theme.glass.subtle.border}`,
  padding: "28px",
  overflow: "hidden",
};

const topAccent: CSSProperties = {
  position: "absolute",
  top: 0,
  left: 0,
  right: 0,
  height: "2px",
  background: theme.gradients.ctaPrimary,
  opacity: 0.6,
};

const iconBadge: CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  width: "40px",
  height: "40px",
  borderRadius: theme.radius.md,
  background: theme.colors.chromeSoft,
  color: theme.colors.chrome,
  marginBottom: "18px",
};

const titleStyle: CSSProperties = {
  margin: "0 0 14px 0",
  fontSize: "1.3rem",
  fontFamily: sansFont,
  color: theme.colors.textPrimary,
  lineHeight: 1.4,
};

const textStyle: CSSProperties = {
  margin: 0,
  color: theme.colors.textSecondary,
  lineHeight: 1.85,
  fontSize: "1.02rem",
};
