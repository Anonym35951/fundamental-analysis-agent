import { useMemo } from "react";
import { theme } from "./theme";
import { useIsMobile } from "../../hooks/useMediaQuery";

type Variant = "hero" | "subtle" | "minimal";

type AmbientBackgroundProps = {
  variant?: Variant;
};

/** Cheap, pure-CSS/SVG decorative backdrop for the monochrome theme: a base
 * white/grey glow wash, a couple of glossy "chrome ribbon" blobs (inspired by
 * dark glossy 3D wave/ribbon reference art), and — on the "hero" variant only
 * — a thin vertical light-beam scan-line. No canvas/WebGL. Mounted once per
 * layout (not per page) so route changes within the same layout don't re-pay
 * the filter/paint cost. Always `pointer-events: none`. */
const NOISE_BG =
  "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")";

export default function AmbientBackground({ variant = "subtle" }: AmbientBackgroundProps) {
  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  }, []);
  // Continuously animating multiple large blur(40-130px)-filtered layers is
  // a known iOS Safari performance cliff (forces repeated GPU recompositing
  // of huge raster layers every frame) — reported live as a freeze/hang when
  // this component mounts fresh on a layout switch (e.g. Landing -> Login).
  // Same treatment as the reduced-motion case: keep the glow/ribbons visible
  // (one-time paint cost) but drop the continuous drift animation on mobile.
  const isMobile = useIsMobile();
  const disableContinuousAnimation = prefersReducedMotion || isMobile;

  const showGlow = variant !== "minimal";
  const showRibbons = variant !== "minimal";
  const showScanLine = variant === "hero" && !isMobile;
  const grainOpacity = variant === "hero" ? 0.045 : variant === "subtle" ? 0.04 : 0.03;
  const ribbonCount = variant === "hero" ? 3 : 2;

  return (
    <div
      aria-hidden="true"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 0,
        pointerEvents: "none",
        overflow: "hidden",
      }}
    >
      {showGlow && (
        <>
          <div
            style={{
              position: "absolute",
              top: "-10%",
              left: "-8%",
              width: variant === "hero" ? "640px" : "420px",
              height: variant === "hero" ? "640px" : "420px",
              background: theme.glow.primary,
              filter: `blur(${theme.glow.blurPx})`,
              opacity: variant === "hero" ? 0.9 : 0.5,
              animation: disableContinuousAnimation ? undefined : "ambient-drift-a 28s ease-in-out infinite",
            }}
          />
          <div
            style={{
              position: "absolute",
              top: "8%",
              right: "-10%",
              width: variant === "hero" ? "520px" : "340px",
              height: variant === "hero" ? "520px" : "340px",
              background: theme.glow.secondary,
              filter: `blur(${theme.glow.blurPx})`,
              opacity: variant === "hero" ? 0.7 : 0.35,
              animation: disableContinuousAnimation ? undefined : "ambient-drift-b 32s ease-in-out infinite",
            }}
          />
        </>
      )}

      {showRibbons && (
        <>
          <div
            style={{
              position: "absolute",
              top: "12%",
              left: "-20%",
              width: "140vw",
              height: "48vh",
              background: theme.gradients.chromeRibbon,
              clipPath: "ellipse(58% 38% at 50% 50%)",
              filter: "blur(50px)",
              opacity: variant === "hero" ? 0.5 : 0.3,
              transform: "translateY(calc(var(--scroll-progress, 0) * -30px)) rotate(-8deg)",
              animation: disableContinuousAnimation ? undefined : "ribbon-drift-a 26s ease-in-out infinite",
            }}
          />
          <div
            style={{
              position: "absolute",
              top: "42%",
              right: "-25%",
              width: "150vw",
              height: "52vh",
              background: theme.gradients.chromeRibbon,
              clipPath: "ellipse(55% 35% at 50% 50%)",
              filter: "blur(58px)",
              opacity: variant === "hero" ? 0.4 : 0.22,
              transform: "translateY(calc(var(--scroll-progress, 0) * 24px)) rotate(10deg)",
              animation: disableContinuousAnimation ? undefined : "ribbon-drift-b 34s ease-in-out infinite",
            }}
          />
          {ribbonCount > 2 && (
            <div
              style={{
                position: "absolute",
                top: "68%",
                left: "-15%",
                width: "130vw",
                height: "40vh",
                background: theme.gradients.chromeRibbon,
                clipPath: "ellipse(50% 32% at 50% 50%)",
                filter: "blur(46px)",
                opacity: 0.28,
                transform: "translateY(calc(var(--scroll-progress, 0) * -18px)) rotate(-3deg)",
                animation: disableContinuousAnimation ? undefined : "ribbon-drift-c 30s ease-in-out infinite",
              }}
            />
          )}
        </>
      )}

      {showScanLine && !prefersReducedMotion && (
        <div
          style={{
            position: "absolute",
            top: 0,
            left: "10%",
            width: "1px",
            height: "55%",
            background: theme.glow.scanLine,
            mixBlendMode: "screen",
            animation: "scan-line-fall 14s ease-in-out infinite",
          }}
        />
      )}

      <div
        style={{
          position: "absolute",
          inset: 0,
          backgroundImage: NOISE_BG,
          backgroundRepeat: "repeat",
          opacity: grainOpacity,
          mixBlendMode: "overlay",
        }}
      />
    </div>
  );
}
