import type { ReactNode } from "react";
import { useState } from "react";
import { theme } from "../ui/theme";
import { useIsTablet } from "../../hooks/useMediaQuery";

export type HeroRailItem = {
  id: string;
  icon: ReactNode;
  label: string;
};

type HeroIconRailProps = {
  items: HeroRailItem[];
  activeId: string;
  onChange: (id: string) => void;
};

/** Decorative-but-interactive vertical icon rail, scoped to the hero only.
 * Clicking an icon sets the same active id used by `AudienceTabs`. Hidden
 * below the tablet breakpoint — on narrow screens the bottom `AudienceTabs`
 * segmented control is the primary way to switch, so the rail would just be
 * redundant chrome rather than a missing feature. */
export default function HeroIconRail({ items, activeId, onChange }: HeroIconRailProps) {
  const isTablet = useIsTablet();
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  if (isTablet) return null;

  return (
    <nav aria-label="Analysearten (Schnellzugriff)" style={rail}>
      {items.map((item) => {
        const isActive = item.id === activeId;
        const isHovered = item.id === hoveredId;
        return (
          <button
            key={item.id}
            type="button"
            onClick={() => onChange(item.id)}
            onMouseEnter={() => setHoveredId(item.id)}
            onMouseLeave={() => setHoveredId(null)}
            aria-label={item.label}
            aria-pressed={isActive}
            title={item.label}
            style={{
              ...railButton,
              ...(isActive ? railButtonActive : {}),
              ...(!isActive && isHovered ? railButtonHover : {}),
            }}
          >
            {item.icon}
          </button>
        );
      })}
    </nav>
  );
}

const rail: React.CSSProperties = {
  position: "absolute",
  left: "8px",
  top: "50%",
  transform: "translateY(-50%)",
  display: "flex",
  flexDirection: "column",
  gap: "10px",
  padding: "10px",
  borderRadius: theme.radius.pill,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  zIndex: 2,
};

const railButton: React.CSSProperties = {
  width: "38px",
  height: "38px",
  borderRadius: theme.radius.pill,
  border: "1px solid transparent",
  background: "transparent",
  color: theme.colors.textMuted,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  cursor: "pointer",
  transition: `background ${theme.motion.fast} ${theme.motion.easing}, color ${theme.motion.fast} ${theme.motion.easing}`,
};

const railButtonActive: React.CSSProperties = {
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
};

const railButtonHover: React.CSSProperties = {
  background: "rgba(255,255,255,0.06)",
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
};
