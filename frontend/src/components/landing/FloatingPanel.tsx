import type { CSSProperties, ReactNode } from "react";
import { useMemo } from "react";
import { motion } from "framer-motion";
import { useIsTablet } from "../../hooks/useMediaQuery";

type FloatingPanelProps = {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Distinguishes instances on desktop so multiple panels don't float in
   * lockstep — ignored on tablet/mobile, where all panels pulse in sync
   * instead (see below). */
  seed?: number;
};

const PULSE_TRANSITION = { duration: 7, repeat: Infinity, ease: "easeInOut" } as const;

/** Thin wrapper applying a slow ambient animation to otherwise-static glass
 * panels. Desktop: independent per-instance up/down float (derived from
 * `seed`, unchanged). Tablet/mobile: panels instead pulse — a synchronized
 * scale/opacity "breathing" in and out of the plane, since independent
 * floating reads as restless at narrower widths where panels stack closer
 * together. Disabled entirely under `prefers-reduced-motion`. */
export default function FloatingPanel({ children, className, style, seed = 0 }: FloatingPanelProps) {
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  }, []);
  const isTablet = useIsTablet();

  if (prefersReducedMotion) {
    return (
      <div className={className} style={style}>
        {children}
      </div>
    );
  }

  if (isTablet) {
    return (
      <motion.div
        className={className}
        style={style}
        animate={{ scale: [1, 1.018, 1], opacity: [1, 0.94, 1] }}
        transition={PULSE_TRANSITION}
      >
        {children}
      </motion.div>
    );
  }

  const amplitude = 8 + (seed % 3) * 3;
  const duration = 5.5 + (seed % 4) * 0.8;
  const delay = (seed % 5) * 0.35;

  return (
    <motion.div
      className={className}
      style={style}
      animate={{ y: [0, -amplitude, 0, amplitude * 0.6, 0] }}
      transition={{
        duration,
        delay,
        repeat: Infinity,
        ease: "easeInOut",
      }}
    >
      {children}
    </motion.div>
  );
}
