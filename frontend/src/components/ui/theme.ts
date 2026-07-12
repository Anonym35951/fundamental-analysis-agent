import { useThemeMode } from "./useThemeMode";

/** Shared design tokens for the monochrome / chrome visual system.
 * Color values resolve through CSS custom properties so the same token
 * names work in both the dark (default/original) and light theme — see
 * the `:root[data-theme="..."]` blocks in `index.css` for the actual
 * values. Never hardcode hex/rgba literals in components — restyle via
 * these tokens only. */
export const theme = {
  colors: {
    black: "var(--color-black)",
    bgGradientStart: "var(--color-bg-gradient-start)",
    bgGradientEnd: "var(--color-bg-gradient-end)",
    panel: "var(--color-panel)",
    panelAlt: "var(--color-panel-alt)",
    border: "var(--color-border)",
    borderSubtle: "var(--color-border-subtle)",
    textPrimary: "var(--color-text-primary)",
    textSecondary: "var(--color-text-secondary)",
    textMuted: "var(--color-text-muted)",
    /** Neutral metallic accent — replaces the old blue/gold brand accents.
     * Used for emphasis, active states, and focus rings, never for
     * "branding" the way accent/accent2 used to. */
    chrome: "var(--color-chrome)",
    chromeStrong: "var(--color-chrome-strong)",
    chromeSoft: "var(--color-chrome-soft)",
    chromeBorder: "var(--color-chrome-border)",
    /** Text/icon color for content sitting on top of `chromeStrong` or
     * `gradients.ctaPrimary` surfaces — those surfaces invert between
     * themes (white-on-dark vs. black-on-light), so the contrasting
     * foreground color must invert with them. */
    onChrome: "var(--color-on-chrome)",
    /** The only color exception: semantic state, desaturated slightly so it
     * still reads as quiet on near-black surfaces. */
    success: "var(--color-success)",
    successSoft: "var(--color-success-soft)",
    successBorder: "var(--color-success-border)",
    danger: "var(--color-danger)",
    dangerSoft: "var(--color-danger-soft)",
    dangerBorder: "var(--color-danger-border)",
    /** Standalone "good"/"bad" status text (e.g. on top of `successSoft`/
     * `dangerSoft` badges) — deliberately distinct from `success`/`danger`
     * since dark mode keeps the original pale mint/pink tone (designed for
     * dark surfaces) while light mode needs a noticeably darker, more
     * saturated shade to stay readable on light surfaces. */
    successText: "var(--color-success-text)",
    dangerText: "var(--color-danger-text)",
    /** True near-black (dark) / near-white (light), used for the
     * marketing/auth surfaces (landing, login) which sit deeper than the
     * app-shell gradient to read as more premium. */
    bgDeep: "var(--color-bg-deep)",
    bgDeepAlt: "var(--color-bg-deep-alt)",
    /** Soft glow used behind light text/elements on deep surfaces — dark in
     * dark mode, light in light mode (the intro slogan's text-shadow). */
    glowSoft: "var(--color-glow-soft)",
  },
  /** Admin-only accent palette (charts/KPI differentiation) — NOT used
   * outside AdminDashboardPage/admin components. The rest of the product
   * stays monochrome/chrome per the established brand system; this is a
   * deliberate, scoped exception for the private admin dashboard's
   * data-viz richness. */
  adminAccent: {
    blue: "var(--color-admin-accent-blue)",
    purple: "var(--color-admin-accent-purple)",
    amber: "var(--color-admin-accent-amber)",
    teal: "var(--color-admin-accent-teal)",
    rose: "var(--color-admin-accent-rose)",
  },
  radius: {
    sm: "10px",
    md: "16px",
    lg: "24px",
    pill: "999px",
  },
  spacing: (factor: number) => `${factor * 4}px`,
  /** Glassmorphism tokens — frosted surfaces tuned darker/higher-contrast in
   * dark mode and softer-shadowed in light mode, so they read as premium
   * rather than milky in either theme. */
  glass: {
    background: "var(--glass-background)",
    backgroundStrong: "var(--glass-background-strong)",
    border: "var(--glass-border)",
    blur: "16px",
    shadow: "var(--glass-shadow)",
    /** Lighter tier for large background panels where a strong frosted look
     * would compete with foreground content. */
    subtle: {
      background: "var(--glass-subtle-background)",
      border: "var(--glass-subtle-border)",
      blur: "10px",
    },
    /** Heavier tier for hero-level / modal-level surfaces that should read as
     * the most prominent glass element on a page. */
    elevated: {
      background: "var(--glass-elevated-background)",
      border: "var(--glass-elevated-border)",
      blur: "24px",
      shadow: "var(--glass-elevated-shadow)",
    },
  },
  gradients: {
    /** Soft halo behind headlines — no color. */
    heroBg: "var(--gradient-hero-bg)",
    /** Brushed chrome/silver sweep — replaces the old blue→gold CTA gradient. */
    ctaPrimary: "var(--gradient-cta-primary)",
    /** Glossy specular highlight overlay for cards — kept neutral. */
    cardSheen: "var(--gradient-card-sheen)",
    /** Soft diagonal sheen band used by the glossy ribbon/wave background
     * shapes (AmbientBackground, decorative hero accents). */
    chromeRibbon: "var(--gradient-chrome-ribbon)",
  },
  /** Soft radial glow + light-beam tokens for the ambient background layer —
   * kept separate from `glass` since these are backdrop decoration, not
   * surface styling. Neutral white/grey in dark mode, soft shadow tones in
   * light mode, no blue/gold. */
  glow: {
    primary: "var(--glow-primary)",
    secondary: "var(--glow-secondary)",
    blurPx: "130px",
    /** Thin moving light-beam/scan-line bar used on the "hero" ambient
     * background variant. */
    scanLine: "var(--glow-scan-line)",
  },
  typography: {
    /** Italic display face reserved for a single accent word in hero
     * headlines — rendered in chrome/white instead of gold now. */
    accentFont: '"Instrument Serif", Georgia, serif',
  },
  /** Centralized animation durations/easing so transitions stay consistent
   * across tab switches, detail expand/collapse, progress bars, etc. */
  motion: {
    fast: "150ms",
    base: "250ms",
    slow: "400ms",
    easing: "cubic-bezier(0.4, 0, 0.2, 1)",
    /** Shared framer-motion spring config for hover/tap physics. */
    spring: { type: "spring", stiffness: 260, damping: 24 } as const,
    /** Base per-item delay (seconds) for staggered reveal sequences. */
    stagger: 0.06,
  },
} as const;

/** Greyscale ramp + grid/axis colors for multi-series charts (recharts) — no
 * decorative rainbow hues in a monochrome system. These are mode-keyed real
 * hex/rgba values (not CSS var() strings) because recharts consumes them as
 * JS prop values rendered straight onto SVG presentation attributes
 * (`stroke`, `fill`), which — unlike the `style` attribute — do not resolve
 * CSS custom properties. Use `useChartTokens()` in chart components instead
 * of reading `theme.chart` directly. */
const chartTokensByMode = {
  dark: {
    series: ["#f5f5f7", "#b0b0b6", "#8e8e93", "#5c5c60", "#34d399", "#f87171"],
    grid: "rgba(255, 255, 255, 0.1)",
    axis: "#8e8e93",
  },
  light: {
    series: ["#1c1c1e", "#52525b", "#71717a", "#a1a1aa", "#16a34a", "#dc2626"],
    grid: "rgba(0, 0, 0, 0.1)",
    axis: "#71717a",
  },
} as const;

export function chartTokens(mode: "light" | "dark") {
  return chartTokensByMode[mode];
}

/** Convenience hook for chart components: returns the mode-correct chart
 * colors (series/grid/axis), since recharts can't resolve CSS var() in its
 * SVG props. */
export function useChartTokens() {
  const { mode } = useThemeMode();
  return chartTokens(mode);
}

/** Richer 7-color series for the admin dashboard's charts only (colorful
 * redesign) — kept as a separate, parallel token set so `useChartTokens()`
 * keeps returning the greyscale palette for any non-admin chart usage.
 * Real hex/rgba values for the same recharts-SVG-prop reason as
 * `chartTokensByMode` above. */
const adminChartTokensByMode = {
  dark: {
    series: ["#f5f5f7", "#60a5fa", "#c084fc", "#fbbf24", "#2dd4bf", "#fb7185", "#34d399"],
    grid: "rgba(255, 255, 255, 0.1)",
    axis: "#8e8e93",
  },
  light: {
    series: ["#1c1c1e", "#2563eb", "#9333ea", "#d97706", "#0d9488", "#e11d48", "#16a34a"],
    grid: "rgba(0, 0, 0, 0.1)",
    axis: "#71717a",
  },
} as const;

/** Convenience hook for the admin dashboard's charts specifically — richer
 * accent palette than `useChartTokens()`, scoped to AdminDashboardPage. */
export function useAdminChartTokens() {
  const { mode } = useThemeMode();
  return adminChartTokensByMode[mode];
}
