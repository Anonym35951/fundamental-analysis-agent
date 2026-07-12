import { useEffect, useState } from "react";

/** Single source of truth for the app's three breakpoint thresholds
 * (RESPONSIVE.md R-P2-1) — previously duplicated as inline magic numbers at
 * every useMediaQuery callsite. */
export const BREAKPOINTS = {
  sm: 640,
  md: 768,
  lg: 1024,
} as const;

export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(
    () => typeof window !== "undefined" && window.matchMedia(query).matches
  );

  useEffect(() => {
    const mediaQueryList = window.matchMedia(query);
    const listener = (event: MediaQueryListEvent) => setMatches(event.matches);

    // Nicht redundant zum useState-Initializer oben: der läuft nur einmal
    // beim Mount, dieser Sync greift, wenn `query` sich zwischen Renders
    // ändert (das [query]-Dependency-Array unten) - ohne ihn würde `matches`
    // bis zur nächsten "change"-Media-Query-Änderung den alten Query-Wert
    // zeigen (LAUNCH_AUDIT.md P2-10, legitimer Sync-zu-externem-Zustand-Fall).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setMatches(mediaQueryList.matches);
    mediaQueryList.addEventListener("change", listener);

    return () => mediaQueryList.removeEventListener("change", listener);
  }, [query]);

  return matches;
}

/** Narrow phones (portrait) and below — the "sm" tier in RESPONSIVE.md's
 * breakpoint table. */
export function useIsNarrow(): boolean {
  return useMediaQuery(`(max-width: ${BREAKPOINTS.sm}px)`);
}

export function useIsMobile(): boolean {
  return useMediaQuery(`(max-width: ${BREAKPOINTS.md}px)`);
}

export function useIsTablet(): boolean {
  return useMediaQuery(`(max-width: ${BREAKPOINTS.lg}px)`);
}
