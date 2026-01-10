import { useEffect, useMemo, useState } from "react";

export default function Background() {
  const [spot, setSpot] = useState({ x: 50, y: 25 }); // in %

  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  }, []);

  useEffect(() => {
    if (prefersReducedMotion) return;

    let raf = 0;

    const onScroll = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(() => {
        const doc = document.documentElement;

        const scrollTop = window.scrollY || doc.scrollTop || 0;
        const scrollHeight = doc.scrollHeight - window.innerHeight;

        const t = scrollHeight > 0 ? scrollTop / scrollHeight : 0;
        const y = 15 + t * 80; // 15% .. 95%
        const x = 50 + Math.sin(t * Math.PI * 2) * 18;

        setSpot({
          x: Math.max(8, Math.min(92, x)),
          y: Math.max(8, Math.min(92, y)),
        });
      });
    };

    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [prefersReducedMotion]);

  return (
    <>
      {/* Base */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: -5,
          background:
            "linear-gradient(180deg, #090a0d 0%, #111218 60%, #090a0d 100%)",
        }}
      />

      {/* Scroll-follow Ambient Spotlight (breiter & softer) */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: -4,
          pointerEvents: "none",
          background: `
            radial-gradient(1200px 700px at ${spot.x}% ${spot.y}%,
              rgba(255,255,255,0.10),
              rgba(255,255,255,0.045) 40%,
              rgba(255,255,255,0.02) 60%,
              transparent 80%
            )
          `,
          transition: prefersReducedMotion ? undefined : "background 140ms linear",
          filter: "blur(22px)",
          opacity: 0.85,
        }}
      />

      {/* Slow drifting light pools */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: -3,
          pointerEvents: "none",
          filter: "blur(46px)",
          opacity: 0.6,
          background:
            "radial-gradient(600px 420px at 18% 30%, rgba(255,255,255,0.06), transparent 70%)," +
            "radial-gradient(540px 420px at 82% 36%, rgba(255,255,255,0.045), transparent 72%)," +
            "radial-gradient(680px 520px at 55% 85%, rgba(255,255,255,0.035), transparent 72%)",
          animation: prefersReducedMotion
            ? undefined
            : "bgDrift 24s ease-in-out infinite alternate",
        }}
      />

      {/* Vignette */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: -2,
          pointerEvents: "none",
          background:
            "radial-gradient(circle at 50% 40%, transparent 0%, rgba(0,0,0,0.55) 78%, rgba(0,0,0,0.75) 100%)",
        }}
      />

      {/* Noise */}
      <div
        style={{
          position: "fixed",
          inset: 0,
          zIndex: -1,
          pointerEvents: "none",
          opacity: 0.08,
          backgroundImage:
            "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.8' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='180' height='180' filter='url(%23n)' opacity='.35'/%3E%3C/svg%3E\")",
        }}
      />

      <style>
        {`
          @keyframes bgDrift {
            0%   { transform: translate3d(-10px, -12px, 0) scale(1.02); }
            50%  { transform: translate3d(12px, 10px, 0) scale(1.04); }
            100% { transform: translate3d(-6px, 16px, 0) scale(1.03); }
          }
        `}
      </style>
    </>
  );
}