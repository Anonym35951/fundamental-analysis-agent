import { useMemo, useState, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { theme } from "./theme";

type StackedCardsProps<T> = {
  items: T[];
  renderCard: (item: T, index: number, isExpanded: boolean) => ReactNode;
  getKey: (item: T, index: number) => string | number;
  /** How many cards peek out behind the front card when collapsed. */
  maxPeek?: number;
  /** Px vertical offset between each peeking card. */
  peekOffset?: number;
  /** Px width inset per peeking card, for the "behind" depth illusion. */
  peekInset?: number;
  expanded?: boolean;
  onExpandedChange?: (expanded: boolean) => void;
  expandLabel?: (hiddenCount: number) => string;
  emptyState?: ReactNode;
};

/** Generic, presentational-only "peekable card stack" — front card fully
 * visible, a handful of partial slivers peeking out behind it (mirrors the
 * iOS "My Reservations" overlapping-cards pattern). Click/tap fans the stack
 * out into a full vertical list. Zero knowledge of caller data shapes — all
 * content rendering is delegated to `renderCard`, so this is purely a layout
 * wrapper around whatever cards the caller already renders elsewhere. */
export default function StackedCards<T>({
  items,
  renderCard,
  getKey,
  maxPeek = 3,
  peekOffset = 14,
  peekInset = 10,
  expanded: expandedProp,
  onExpandedChange,
  expandLabel,
  emptyState,
}: StackedCardsProps<T>) {
  const [internalExpanded, setInternalExpanded] = useState(false);
  const expanded = expandedProp ?? internalExpanded;

  const prefersReducedMotion = useMemo(() => {
    if (typeof window === "undefined") return false;
    return window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
  }, []);

  function setExpanded(next: boolean) {
    setInternalExpanded(next);
    onExpandedChange?.(next);
  }

  if (items.length === 0) {
    return emptyState ? <>{emptyState}</> : null;
  }

  if (items.length === 1) {
    return <div>{renderCard(items[0], 0, true)}</div>;
  }

  if (expanded) {
    return (
      <div style={{ display: "flex", flexDirection: "column", gap: theme.spacing(3) }}>
        <button type="button" onClick={() => setExpanded(false)} style={collapseButton}>
          <ChevronDown size={14} style={{ transform: "rotate(180deg)" }} />
          Stapel schließen
        </button>

        <AnimatePresence initial={false}>
          {items.map((item, index) => (
            <motion.div
              key={getKey(item, index)}
              layout={!prefersReducedMotion}
              initial={prefersReducedMotion ? undefined : { opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: prefersReducedMotion ? 0 : 0.22, ease: [0.4, 0, 0.2, 1] }}
            >
              {renderCard(item, index, true)}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    );
  }

  const visiblePeeks = Math.min(maxPeek, items.length - 1);
  const hiddenCount = items.length - 1;

  return (
    <div
      style={{
        position: "relative",
        paddingBottom: `${visiblePeeks * peekOffset + 8}px`,
      }}
    >
      {Array.from({ length: visiblePeeks })
        .map((_, i) => visiblePeeks - i)
        .map((depth) => {
          const item = items[depth];
          if (!item) return null;
          return (
            <div
              key={getKey(item, depth)}
              aria-hidden="true"
              style={{
                position: "absolute",
                left: `${depth * peekInset}px`,
                right: `${depth * peekInset}px`,
                top: `${depth * peekOffset}px`,
                height: "28px",
                borderRadius: theme.radius.lg,
                background: theme.colors.panelAlt,
                border: `1px solid ${theme.glass.subtle.border}`,
                boxShadow: "0 10px 24px rgba(0, 0, 0, 0.18)",
              }}
            />
          );
        })}

      <button
        type="button"
        onClick={() => setExpanded(true)}
        style={{
          position: "relative",
          display: "block",
          width: "100%",
          textAlign: "left",
          background: "none",
          border: "none",
          padding: 0,
          cursor: "pointer",
        }}
      >
        {renderCard(items[0], 0, false)}
      </button>

      {hiddenCount > 0 && (
        <button type="button" onClick={() => setExpanded(true)} style={expandHint}>
          {expandLabel ? expandLabel(hiddenCount) : `${hiddenCount} weitere anzeigen`}
          <ChevronDown size={13} />
        </button>
      )}
    </div>
  );
}

const collapseButton = {
  display: "inline-flex",
  alignItems: "center",
  gap: "6px",
  alignSelf: "flex-start" as const,
  padding: "6px 12px",
  borderRadius: theme.radius.pill,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textSecondary,
  fontSize: "0.8rem",
  fontWeight: 700,
  cursor: "pointer",
};

const expandHint = {
  position: "absolute" as const,
  bottom: 0,
  right: "8px",
  display: "inline-flex",
  alignItems: "center",
  gap: "4px",
  padding: "4px 10px",
  borderRadius: theme.radius.pill,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
  fontSize: "0.74rem",
  fontWeight: 700,
  cursor: "pointer",
  zIndex: 5,
};
