import type { ReactNode } from "react";
import { useEffect, useRef, useState } from "react";
import { theme } from "../ui/theme";

export type AudienceTab = {
  id: string;
  label: string;
  description: string;
  /** Short, bolded value statement shown above the description — the
   * benefit a user gets from this mode, read before the workflow detail. */
  benefit: string;
  icon: ReactNode;
};

type AudienceTabsProps = {
  tabs: AudienceTab[];
  activeId: string;
  onChange: (id: string) => void;
};

/** Bottom segmented control over real analysis modes — switching updates
 * whatever content the caller derives from `activeId` (the hero's central
 * panel). Horizontally scrollable so it degrades gracefully on narrow
 * screens instead of wrapping awkwardly. */
export default function AudienceTabs({ tabs, activeId, onChange }: AudienceTabsProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const navRef = useRef<HTMLElement | null>(null);
  // Zeigt, ob es überhaupt noch etwas zu scrollen gibt (und in welche
  // Richtung) — der letzte Chip wurde sonst ohne jede Andeutung angeschnitten
  // (RESPONSIVE.md R-P1-8). Reine Sichtbarkeits-Info, kein Scroll-Handling.
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  useEffect(() => {
    const nav = navRef.current;
    if (!nav) return;

    function updateFades() {
      setCanScrollLeft(nav!.scrollLeft > 2);
      setCanScrollRight(nav!.scrollLeft + nav!.clientWidth < nav!.scrollWidth - 2);
    }

    updateFades();
    nav.addEventListener("scroll", updateFades, { passive: true });
    const resizeObserver = new ResizeObserver(updateFades);
    resizeObserver.observe(nav);
    return () => {
      nav.removeEventListener("scroll", updateFades);
      resizeObserver.disconnect();
    };
  }, [tabs]);

  // Mask statt eines farbig überlagerten Divs: ein Overlay in der eigenen
  // (sehr blassen) Glass-Hintergrundfarbe war auf der Live-Probe praktisch
  // unsichtbar, da die Deckkraft des Tokens selbst schon minimal ist. Eine
  // Opazitäts-Maske auf dem Scroll-Container blendet den echten Inhalt aus,
  // unabhängig von jeder Hintergrundfarbe (gleiche Technik wie
  // ComparePivotTable, RESPONSIVE.md R-P1-8).
  const maskStops: string[] = [];
  if (canScrollLeft) maskStops.push("transparent", "black 20px");
  else maskStops.push("black 0");
  if (canScrollRight) maskStops.push("black calc(100% - 20px)", "transparent");
  else maskStops.push("black 100%");
  const fadeMask =
    canScrollLeft || canScrollRight ? `linear-gradient(to right, ${maskStops.join(", ")})` : undefined;

  return (
    <nav
      ref={navRef}
      aria-label="Analysearten"
      style={{ ...wrapper, maskImage: fadeMask, WebkitMaskImage: fadeMask }}
    >
      {tabs.map((tab) => {
        const isActive = tab.id === activeId;
        const isHovered = tab.id === hoveredId;
        return (
          <button
            key={tab.id}
            type="button"
            onClick={() => onChange(tab.id)}
            onMouseEnter={() => setHoveredId(tab.id)}
            onMouseLeave={() => setHoveredId(null)}
            aria-pressed={isActive}
            style={{
              ...tabButton,
              ...(isActive ? tabButtonActive : {}),
              ...(!isActive && isHovered ? tabButtonHover : {}),
            }}
          >
            <span style={iconWrap}>{tab.icon}</span>
            {tab.label}
          </button>
        );
      })}
    </nav>
  );
}

const wrapper: React.CSSProperties = {
  display: "flex",
  gap: "8px",
  padding: "6px",
  borderRadius: theme.radius.pill,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  overflowX: "auto",
  WebkitOverflowScrolling: "touch",
  overscrollBehaviorX: "contain",
  maxWidth: "100%",
};

const tabButton: React.CSSProperties = {
  flexShrink: 0,
  display: "inline-flex",
  alignItems: "center",
  gap: "8px",
  padding: "10px 16px",
  borderRadius: theme.radius.pill,
  border: "1px solid transparent",
  background: "transparent",
  color: theme.colors.textSecondary,
  fontWeight: 600,
  fontSize: "0.88rem",
  cursor: "pointer",
  transition: `background ${theme.motion.fast} ${theme.motion.easing}, color ${theme.motion.fast} ${theme.motion.easing}`,
  whiteSpace: "nowrap",
};

const tabButtonActive: React.CSSProperties = {
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
};

const tabButtonHover: React.CSSProperties = {
  background: "rgba(255,255,255,0.06)",
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
};

const iconWrap: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
};
