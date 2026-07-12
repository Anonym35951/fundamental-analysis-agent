import { useEffect, useRef } from "react";
import { useThemeMode } from "../ui/useThemeMode";

type Particle = {
  /** Normalized horizontal position within the beam (-1..1), independent of
   * pixel width so resize doesn't require re-seeding. */
  nx: number;
  /** Vertical position in px, owns its own respawn-at-top cycle. */
  y: number;
  speed: number;
  size: number;
  opacity: number;
  /** Phase offset for the horizontal sway once a particle reaches the
   * "ground" band — keeps the wave look from the reference image without
   * every particle swaying in lockstep. */
  swayPhase: number;
  swaySpeed: number;
};

const PARTICLE_COUNT_DESKTOP = 360;
const PARTICLE_COUNT_MOBILE = 180;
/** Extra particles per 1000px of page height beyond a 1200px baseline, so a
 * full-page-height canvas (this now spans the whole LandingPage, not just
 * the hero) doesn't look sparse lower down. */
const PARTICLES_PER_1000PX = 60;
const MAX_PARTICLE_COUNT = 520;

function createParticles(count: number, height: number): Particle[] {
  return Array.from({ length: count }, () => ({
    nx: (Math.random() * 2 - 1) * 0.06 + (Math.random() < 0.5 ? -1 : 1) * Math.random() * 0.9,
    y: Math.random() * height,
    speed: 22 + Math.random() * 30,
    size: 0.6 + Math.random() * 1.6,
    opacity: 0.15 + Math.random() * 0.55,
    swayPhase: Math.random() * Math.PI * 2,
    swaySpeed: 0.4 + Math.random() * 0.6,
  }));
}

type ParticleBeamBackgroundProps = {
  /** Multiplies the computed particle count before the MAX_PARTICLE_COUNT
   * clamp. Defaults to 1 so every existing call site (landing page, auth
   * layout) stays pixel-identical; pass >1 for a denser look on a specific
   * page without touching the shared base counts. */
  densityMultiplier?: number;
};

/** Live, monochrome particle-beam effect inspired by a vertical light-beam
 * reference image: a bright focal stream near the top funnels down into a
 * wider, wavy "ground" band of drifting particles. Canvas 2D only (no
 * WebGL/Three.js) — cheap enough to run continuously, but still genuinely
 * animated rather than a static CSS approximation. Hero-scoped: mounted as
 * one absolutely-positioned layer behind the hero content. */
export default function ParticleBeamBackground({ densityMultiplier = 1 }: ParticleBeamBackgroundProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const { mode } = useThemeMode();
  // Read via a ref inside the animation loop instead of depending on `mode`
  // in the main effect below — that effect owns the particle simulation
  // state and shouldn't restart (resetting positions) on every theme toggle.
  const particleRgbRef = useRef(mode === "light" ? "40, 40, 44" : "245, 245, 247");

  useEffect(() => {
    particleRgbRef.current = mode === "light" ? "40, 40, 44" : "245, 245, 247";
  }, [mode]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;

    let width = 0;
    let height = 0;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    let particles: Particle[] = [];
    let rafId: number | null = null;
    let isVisible = true;
    let lastTime = performance.now();

    function resize() {
      const rect = container!.getBoundingClientRect();
      width = rect.width;
      height = rect.height;
      canvas!.width = width * dpr;
      canvas!.height = height * dpr;
      canvas!.style.width = `${width}px`;
      canvas!.style.height = `${height}px`;
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0);

      const base = width < 640 ? PARTICLE_COUNT_MOBILE : PARTICLE_COUNT_DESKTOP;
      const extraForHeight = Math.max(0, height - 1200) / 1000 * PARTICLES_PER_1000PX;
      const count = Math.min(MAX_PARTICLE_COUNT, Math.round((base + extraForHeight) * densityMultiplier));
      if (particles.length !== count) {
        particles = createParticles(count, height);
      }
    }

    function drawFrame(elapsedSeconds: number) {
      ctx!.clearRect(0, 0, width, height);

      const groundY = height * 0.62;
      const centerX = width / 2;
      const beamHalfWidth = Math.max(60, width * 0.16);

      for (const p of particles) {
        const t = p.y / groundY; // 0 at top focal point, 1 at the ground band
        const funnel = Math.min(1, Math.max(0, t));
        // Particles start tight near the beam center, fan out as they fall.
        const spread = beamHalfWidth * (0.15 + funnel * 1.6);
        let x = centerX + p.nx * spread;

        if (p.y > groundY) {
          // Past the ground line: add a slow horizontal sway for the
          // "flowing wave" look from the reference art.
          const swayAmount = 14 + Math.abs(p.nx) * 18;
          x += Math.sin(elapsedSeconds * p.swaySpeed + p.swayPhase) * swayAmount;
        }

        const brightness = funnel < 1 ? 1 - funnel * 0.4 : 0.55;
        ctx!.beginPath();
        ctx!.fillStyle = `rgba(${particleRgbRef.current}, ${p.opacity * brightness})`;
        ctx!.arc(x, p.y, p.size, 0, Math.PI * 2);
        ctx!.fill();
      }
    }

    function step(now: number) {
      const dt = Math.min(0.05, (now - lastTime) / 1000);
      lastTime = now;
      const elapsedSeconds = now / 1000;

      const groundY = height * 0.62;
      for (const p of particles) {
        // Slows down as it approaches/passes the ground band, like the
        // reference image's particles settling into the wave texture.
        const speedFactor = p.y > groundY ? 0.25 : 1;
        p.y += p.speed * speedFactor * dt;
        if (p.y > height + 10) {
          p.y = -10;
        }
      }

      drawFrame(elapsedSeconds);

      if (isVisible && !prefersReducedMotion) {
        rafId = requestAnimationFrame(step);
      }
    }

    resize();
    if (prefersReducedMotion) {
      drawFrame(0);
    } else {
      lastTime = performance.now();
      rafId = requestAnimationFrame(step);
    }

    const resizeObserver = new ResizeObserver(() => {
      resize();
      if (prefersReducedMotion) drawFrame(0);
    });
    resizeObserver.observe(container);

    const intersectionObserver = new IntersectionObserver(
      ([entry]) => {
        isVisible = entry.isIntersecting;
        if (isVisible && rafId === null && !prefersReducedMotion) {
          lastTime = performance.now();
          rafId = requestAnimationFrame(step);
        }
      },
      { threshold: 0 }
    );
    intersectionObserver.observe(container);

    return () => {
      if (rafId !== null) cancelAnimationFrame(rafId);
      resizeObserver.disconnect();
      intersectionObserver.disconnect();
    };
  }, [densityMultiplier]);

  return (
    <div
      ref={containerRef}
      aria-hidden="true"
      style={{
        position: "absolute",
        inset: 0,
        overflow: "hidden",
        pointerEvents: "none",
        zIndex: 0,
      }}
    >
      <canvas ref={canvasRef} />
    </div>
  );
}
