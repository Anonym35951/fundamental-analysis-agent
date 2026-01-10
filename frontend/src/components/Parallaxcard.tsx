//frontend/src/components/ParallaxCard.tsx
import React, { useMemo, useRef, useState } from "react";

type Props = {
  children: React.ReactNode;
  className?: string;
  style?: React.CSSProperties;
  /** max tilt in deg */
  tilt?: number; // default 6
  /** max translate in px */
  lift?: number; // default 10
};

export default function ParallaxCard({
  children,
  className,
  style,
  tilt = 6,
  lift = 10,
}: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);

  const [isHover, setIsHover] = useState(false);
  const [vars, setVars] = useState({ rx: 0, ry: 0, tx: 0, ty: 0 });

  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  }, []);

  const apply = (next: typeof vars) => {
    setVars(next);
    const el = ref.current;
    if (!el) return;

    el.style.setProperty("--rx", `${next.rx}deg`);
    el.style.setProperty("--ry", `${next.ry}deg`);
    el.style.setProperty("--tx", `${next.tx}px`);
    el.style.setProperty("--ty", `${next.ty}px`);
  };

  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (prefersReducedMotion) return;
    const el = ref.current;
    if (!el) return;

    const rect = el.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;  // 0..1
    const py = (e.clientY - rect.top) / rect.height; // 0..1

    const dx = px - 0.5; // -0.5..0.5
    const dy = py - 0.5;

    const next = {
      ry: dx * tilt * 2,          // rotateY
      rx: -dy * tilt * 2,         // rotateX (invert for natural feel)
      tx: dx * lift * 2,          // translateX
      ty: dy * lift * 2,          // translateY
    };

    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(() => apply(next));
  };

  const onEnter = () => {
    setIsHover(true);
  };

  const onLeave = () => {
    setIsHover(false);
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
    apply({ rx: 0, ry: 0, tx: 0, ty: 0 });
  };

  return (
    <div
      ref={ref}
      className={`parallaxCard ${isHover ? "isHover" : ""} ${className ?? ""}`}
      style={{
        ...style,
        // initial vars (falls vor CSS)
        ["--rx" as any]: "0deg",
        ["--ry" as any]: "0deg",
        ["--tx" as any]: "0px",
        ["--ty" as any]: "0px",
      }}
      onPointerEnter={onEnter}
      onPointerLeave={onLeave}
      onPointerMove={onMove}
    >