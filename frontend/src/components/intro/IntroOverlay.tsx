import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import RingsScene, { FLY_IN_DURATION, FLY_OUT_DURATION, type IntroPhase } from "./RingsScene";
import IntroSlogan from "./IntroSlogan";
import { theme } from "../ui/theme";

const BLACK_HOLD_MS = 300;
const HOLD_MS = 1000;

export type OverlayPhase = "black" | "in" | "hold" | "out" | "done";

type IntroOverlayProps = {
  /** Reports every phase change upward so AppLayout can fade the real app
   * content in exactly in sync with this overlay fading out, instead of the
   * (already fully-rendered) page just popping into view the instant the
   * overlay disappears. */
  onPhaseChange?: (phase: OverlayPhase) => void;
};

/** Fully automatic post-login intro, shown as an overlay on top of the
 * already-mounted app shell (AppLayout renders this conditionally based on
 * a one-time sessionStorage flag — see AppLayout.tsx). No button: rings fly
 * in, hold briefly, then fly back out in reverse while the whole overlay
 * fades, revealing the real app underneath (which has been rendering the
 * entire time, not a placeholder). Self-contained — returns null once its
 * sequence finishes, removing itself from the DOM. */
export default function IntroOverlay({ onPhaseChange }: IntroOverlayProps) {
  const [phase, setPhase] = useState<OverlayPhase>("black");

  useEffect(() => {
    onPhaseChange?.(phase);
  }, [phase, onPhaseChange]);

  useEffect(() => {
    const blackTimer = setTimeout(() => setPhase("in"), BLACK_HOLD_MS);
    return () => clearTimeout(blackTimer);
  }, []);

  useEffect(() => {
    if (phase !== "in") return;
    const timer = setTimeout(() => setPhase("hold"), FLY_IN_DURATION * 1000);
    return () => clearTimeout(timer);
  }, [phase]);

  useEffect(() => {
    if (phase !== "hold") return;
    const timer = setTimeout(() => setPhase("out"), HOLD_MS);
    return () => clearTimeout(timer);
  }, [phase]);

  useEffect(() => {
    if (phase !== "out") return;
    const timer = setTimeout(() => setPhase("done"), FLY_OUT_DURATION * 1000);
    return () => clearTimeout(timer);
  }, [phase]);

  if (phase === "done") return null;

  const ringsPhase: IntroPhase = phase === "black" ? "idle" : phase;
  const sloganVisible = phase === "in" || phase === "hold";

  return (
    // Background and the fade-out animation live on the SAME element here —
    // previously the black background sat on a separate, never-animated
    // outer <div> while only an inner motion.div faded, so the backdrop
    // stayed 100% opaque for the whole "out" phase and only vanished in one
    // instant cut on unmount. Now the whole layer — backdrop included —
    // fades from opaque to transparent over FLY_OUT_DURATION, actually
    // revealing the dashboard crossfading in underneath (see AppLayout.tsx)
    // instead of hiding it until the very end.
    <motion.div
      initial={{ opacity: 1 }}
      animate={{ opacity: phase === "out" ? 0 : 1 }}
      transition={{ duration: FLY_OUT_DURATION, ease: "easeInOut" }}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: theme.colors.bgDeep,
        overflow: "hidden",
        pointerEvents: phase === "out" ? "none" : "auto",
      }}
    >
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: phase === "black" ? 0 : 1 }}
        transition={{ duration: 0.5, ease: "easeInOut" }}
        style={{ position: "absolute", inset: 0 }}
      >
        <RingsScene phase={ringsPhase} />
      </motion.div>

      <IntroSlogan visible={sloganVisible} />
    </motion.div>
  );
}
