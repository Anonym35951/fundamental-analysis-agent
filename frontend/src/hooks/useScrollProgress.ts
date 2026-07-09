import { useEffect } from "react";

/** Writes the document's scroll progress (0..1) to a CSS variable
 * (`--scroll-progress`) on the root element, rAF-throttled, instead of
 * triggering a React re-render on every scroll tick. Consumers (e.g.
 * AmbientBackground) read the variable directly in inline styles via
 * `var(--scroll-progress, 0)` — cheap, no JS subscription needed per
 * component. Call once per layout; not registered at all when the user
 * prefers reduced motion, since the only consumer is decorative parallax. */
export function useScrollProgress(): void {
  useEffect(() => {
    const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
    if (prefersReducedMotion) return;

    const root = document.documentElement;
    let ticking = false;

    function update() {
      ticking = false;
      const max = document.documentElement.scrollHeight - window.innerHeight;
      const progress = max > 0 ? window.scrollY / max : 0;
      root.style.setProperty("--scroll-progress", String(Math.min(1, Math.max(0, progress))));
    }

    function onScroll() {
      if (ticking) return;
      ticking = true;
      requestAnimationFrame(update);
    }

    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);
}
