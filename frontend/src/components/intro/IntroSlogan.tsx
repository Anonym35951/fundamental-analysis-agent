import { motion } from "framer-motion";
import { theme } from "../ui/theme";

type IntroSloganProps = {
  visible: boolean;
};

/** Slow, independent fade for the intro's headline copy — fades in while the
 * rings fly in/hold, and fades back out (mirrored) while they fly back out,
 * rather than just appearing once and staying. */
export default function IntroSlogan({ visible }: IntroSloganProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={visible ? { opacity: 1, y: 0 } : { opacity: 0, y: 8 }}
      transition={{ duration: 2.4, ease: "easeInOut", delay: visible ? 0.3 : 0 }}
      style={{
        position: "absolute",
        top: "14%",
        left: "50%",
        translate: "-50% 0",
        textAlign: "center",
        width: "min(90vw, 680px)",
        color: theme.colors.textPrimary,
        textShadow: `0 0 24px ${theme.colors.glowSoft}`,
      }}
    >
      <p
        style={{
          margin: 0,
          fontSize: "1.9rem",
          fontWeight: 600,
          letterSpacing: "0.01em",
          lineHeight: 1.5,
        }}
      >
        Beginne deine Analyse jetzt.
        <br />
        Deine Reise startet hier.
      </p>
    </motion.div>
  );
}
