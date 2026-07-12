import { useIsNarrow } from "../../hooks/useMediaQuery";
import { theme } from "../ui/theme";

type Props = {
  groupNames: string[];
  activeGroup: string;
  onSelect: (name: string) => void;
  counts: Record<string, number>;
};

/** Vertical nav rail for the dossier's category list — scales to any number
 * of categories without the re-flowing wrap of a pill row, and reads like a
 * Notion/Arc sidebar rather than a tab strip. Collapses to a horizontal
 * segmented control under ~640px where a side column would crowd the page. */
export default function DossierCategoryRail({ groupNames, activeGroup, onSelect, counts }: Props) {
  const isNarrow = useIsNarrow();

  return (
    <div
      style={{
        display: "flex",
        flexDirection: isNarrow ? "row" : "column",
        flexWrap: isNarrow ? ("wrap" as const) : ("nowrap" as const),
        gap: "6px",
      }}
    >
      {groupNames.map((name) => {
        const isActive = activeGroup === name;

        return (
          <button
            key={name}
            type="button"
            onClick={() => onSelect(name)}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              gap: "10px",
              padding: "11px 14px",
              borderRadius: isNarrow ? theme.radius.pill : theme.radius.md,
              border: `1px solid ${isActive ? theme.colors.chromeBorder : "transparent"}`,
              background: isActive ? theme.colors.chromeSoft : "transparent",
              color: isActive ? theme.colors.textPrimary : theme.colors.textSecondary,
              fontWeight: isActive ? 800 : 600,
              fontSize: "0.9rem",
              cursor: "pointer",
              textAlign: "left" as const,
              transition: `background ${theme.motion.base} ${theme.motion.easing}, color ${theme.motion.base} ${theme.motion.easing}`,
              whiteSpace: "nowrap" as const,
            }}
          >
            <span>{name}</span>
            <span
              style={{
                fontSize: "0.74rem",
                fontWeight: 800,
                color: isActive ? theme.colors.chrome : theme.colors.textMuted,
              }}
            >
              {counts[name] ?? 0}
            </span>
          </button>
        );
      })}
    </div>
  );
}
