import type { CSSProperties } from "react";
import { useEffect, useRef } from "react";
import { animate, motion, useMotionValue, useTransform } from "framer-motion";

type AnimatedNumberProps = {
  value: number;
  /** Number of decimal places to render. */
  decimals?: number;
  prefix?: string;
  suffix?: string;
  style?: CSSProperties;
  className?: string;
};

/** Animates a numeric value from its previous render to the new one on
 * change, purely presentational — callers keep passing plain numbers, no
 * change to where the data comes from. */
export default function AnimatedNumber({
  value,
  decimals = 0,
  prefix = "",
  suffix = "",
  style,
  className,
}: AnimatedNumberProps) {
  const motionValue = useMotionValue(value);
  const rounded = useTransform(motionValue, (latest) =>
    `${prefix}${latest.toFixed(decimals)}${suffix}`
  );
  const prevValue = useRef(value);

  useEffect(() => {
    const prefersReducedMotion =
      typeof window !== "undefined" &&
      (window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false);

    if (prefersReducedMotion) {
      motionValue.set(value);
      prevValue.current = value;
      return;
    }

    const controls = animate(prevValue.current, value, {
      duration: 0.6,
      ease: [0.4, 0, 0.2, 1],
      onUpdate: (latest) => motionValue.set(latest),
    });
    prevValue.current = value;
    return () => controls.stop();
  }, [value, motionValue]);

  return (
    <motion.span className={className} style={style}>
      {rounded}
    </motion.span>
  );
}
