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
