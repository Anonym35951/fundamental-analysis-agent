import type { CSSProperties } from "react";
import { motion } from "framer-motion";
import { theme } from "../ui/theme";

type StepCardProps = {
  index: number;
  text: string;
};

const fadeUp = {
  hidden: { opacity: 0, y: 24 },
  visible: { opacity: 1, y: 0 },
};

/** How-it-works numbered step card. A thin connecting line sits behind the
 * row (rendered once by the caller) so steps read as a single flow. */
export default function StepCard({ index, text }: StepCardProps) {
  return (
    <motion.div
      variants={fadeUp}
      transition={{ duration: 0.45, ease: [0.4, 0, 0.2, 1], y: theme.motion.spring, scale: theme.motion.spring }}
      whileHover={{ y: -4, scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      style={card}
    >
      <div style={numberBadge}>{index + 1}</div>
      <div style={textStyle}>{text}</div>
    </motion.div>
  );
}

const card: CSSProperties = {
  position: "relative",
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  borderRadius: theme.radius.md,
  padding: "22px",
};

const numberBadge: CSSProperties = {
  width: "36px",
  height: "36px",
  borderRadius: theme.radius.pill,
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontWeight: 800,
  marginBottom: "18px",
  fontSize: "0.98rem",
};

const textStyle: CSSProperties = {
  fontWeight: 700,
  color: theme.colors.textPrimary,
  lineHeight: 1.65,
  fontSize: "1.08rem",
};
