import type { CSSProperties, ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import { animate, useMotionValue } from "framer-motion";
import { theme } from "./theme";

type ParallaxCardProps = {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Max tilt in degrees. */
  tilt?: number;
  /** Max translate in px. */
  lift?: number;
  /** Continuous idle "breathing" spin on rotateY when not hovered — long
   * readable pause near 0deg, then a quick swing out to the side and back.
   * Reserved for a single hero showcase card (landing page), not the generic
   * hover-tilt cards used elsewhere. */
  idleSpin?: boolean;
};

/** Hero summary card with a subtle 3D tilt-on-hover effect, ported from the
 * experimental "src alt" Parallaxcard prototype into the canonical design
 * system. Reserved for a small number of high-level summary cards per
 * analysis result — applying this to dense data rows/tables would feel
 * gimmicky and hurt scroll performance. */
export default function ParallaxCard({
  children,
  className,
  style,
  tilt = 6,
  lift = 10,
  idleSpin = false,
}: ParallaxCardProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const isHoverRef = useRef(false);
  const [isHover, setIsHover] = useState(false);
  const idleRotation = useMotionValue(0);

  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  }, []);

  function apply(next: { rx: number; ry: number; tx: number; ty: number }) {
    const el = ref.current;
    if (!el) return;
    el.style.setProperty("--rx", `${next.rx}deg`);
    el.style.setProperty("--ry", `${next.ry}deg`);
    el.style.setProperty("--tx", `${next.tx}px`);
    el.style.setProperty("--ty", `${next.ty}px`);
  }

  useEffect(() => {
    if (!idleSpin || prefersReducedMotion) return;

    const controls = animate(idleRotation, [-4, -4, 16, -4, -20, -4], {
      duration: 9,
      times: [0, 0.35, 0.5, 0.65, 0.85, 1],
      ease: "easeInOut",
      repeat: Infinity,
    });

    const unsubscribe = idleRotation.on("change", (value) => {
      if (isHoverRef.current) return;
      const el = ref.current;
      if (!el) return;
      el.style.setProperty("--ry", `${value}deg`);
    });

    return () => {
      controls.stop();
      unsubscribe();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idleSpin, prefersReducedMotion]);

  function handlePointerMove(event: React.PointerEvent<HTMLDivElement>) {
    if (prefersReducedMotion) return;
    const el = ref.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const px = (event.clientX - rect.left) / rect.width;
    const py = (event.clientY - rect.top) / rect.height;
    const dx = px - 0.5;
    const dy = py - 0.5;

    const next = {
      ry: dx * tilt * 2,
      rx: -dy * tilt * 2,
      tx: dx * lift * 2,
      ty: dy * lift * 2,
    };

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => apply(next));
  }

  function handlePointerEnter() {
    isHoverRef.current = true;
    setIsHover(true);
  }

  function handlePointerLeave() {
    isHoverRef.current = false;
    setIsHover(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    if (idleSpin && !prefersReducedMotion) {
      // Let the idle spin resume from wherever it currently is — only the
      // pointer-only properties reset, rotateY keeps animating.
      apply({ rx: 0, ry: idleRotation.get(), tx: 0, ty: 0 });
    } else {
      apply({ rx: 0, ry: 0, tx: 0, ty: 0 });
    }
  }

  return (
    <div
      ref={ref}
      className={className}
      onPointerEnter={handlePointerEnter}
      onPointerLeave={handlePointerLeave}
      onPointerMove={handlePointerMove}
      style={{
        ["--rx" as string]: "0deg",
        ["--ry" as string]: "0deg",
        ["--tx" as string]: "0px",
        ["--ty" as string]: "0px",
        transform:
          "perspective(900px) rotateX(var(--rx)) rotateY(var(--ry)) translate(var(--tx), var(--ty))",
        // While the idle spin drives --ry every frame, a CSS transition on
        // transform would fight that per-frame update and rubber-band — only
        // ease the transform via CSS for pointer-hover interactions.
        transition: prefersReducedMotion
          ? undefined
          : isHover || !idleSpin
            ? `transform ${theme.motion.base} ${theme.motion.easing}, box-shadow ${theme.motion.base} ${theme.motion.easing}`
            : `box-shadow ${theme.motion.base} ${theme.motion.easing}`,
        boxShadow: isHover
          ? "0 28px 70px rgba(0, 0, 0, 0.55)"
          : "0 14px 34px rgba(0, 0, 0, 0.3)",
        background: theme.glass.backgroundStrong,
        border: `1px solid ${theme.glass.border}`,
        backdropFilter: `blur(${theme.glass.blur})`,
        WebkitBackdropFilter: `blur(${theme.glass.blur})`,
        borderRadius: theme.radius.lg,
        padding: "24px",
        color: theme.colors.textPrimary,
        willChange: "transform",
        ...style,
      }}
    >
      {children}
    </div>
  );
}
