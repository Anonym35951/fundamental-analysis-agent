import { useEffect, useId, useRef, useState, type CSSProperties } from "react";
import { createPortal } from "react-dom";
import { AnimatePresence, motion } from "framer-motion";
import { Info } from "lucide-react";
import { theme } from "./theme";
import { getMetricConfig } from "../../config/metricsConfig";

export type InfoTooltipProps = {
  /** Metric key looked up in METRICS_CONFIG. If it has no `description`,
   * this component renders nothing — no icon, no empty popover. */
  metricKey: string;
  /** Icon size in px. */
  size?: number;
  style?: CSSProperties;
};

const POPOVER_WIDTH = 280;
const VIEWPORT_MARGIN = 12;
const CLOSE_DELAY_MS = 100;

/** Small "i" icon that reveals what a metric means and how it's calculated
 * in this codebase. Content is driven entirely by METRICS_CONFIG so every
 * call site just passes a metricKey — no per-call-site conditional wiring.
 * Renders via a portal with viewport-fixed positioning so it isn't clipped
 * by scrollable/overflow-hidden ancestors (e.g. ComparePivotTable's
 * horizontally scrolling wrapper). */
export default function InfoTooltip({ metricKey, size = 14, style }: InfoTooltipProps) {
  const config = getMetricConfig(metricKey);
  const [isOpen, setIsOpen] = useState(false);
  const [coords, setCoords] = useState<{ top: number; left: number; placeAbove: boolean } | null>(null);
  const triggerRef = useRef<HTMLSpanElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const closeTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const popoverId = useId();

  const clearCloseTimer = () => {
    if (closeTimer.current) {
      clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
  };

  const open = () => {
    clearCloseTimer();
    const rect = triggerRef.current?.getBoundingClientRect();
    if (!rect) return;

    const estimatedHeight = 140;
    const placeAbove = rect.bottom + estimatedHeight + VIEWPORT_MARGIN > window.innerHeight;
    const left = Math.min(Math.max(rect.left, VIEWPORT_MARGIN), window.innerWidth - POPOVER_WIDTH - VIEWPORT_MARGIN);
    const top = placeAbove ? rect.top - 6 : rect.bottom + 6;

    setCoords({ top, left, placeAbove });
    setIsOpen(true);
  };

  const scheduleClose = () => {
    clearCloseTimer();
    closeTimer.current = setTimeout(() => setIsOpen(false), CLOSE_DELAY_MS);
  };

  const closeNow = () => {
    clearCloseTimer();
    setIsOpen(false);
  };

  useEffect(() => clearCloseTimer, []);

  useEffect(() => {
    if (!isOpen) return;

    function handlePointerDown(event: PointerEvent) {
      const target = event.target as Node;
      if (triggerRef.current?.contains(target) || popoverRef.current?.contains(target)) return;
      closeNow();
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") closeNow();
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  if (!config?.description) return null;

  return (
    <>
      {/* A real <button> would be more semantically correct, but this icon
          is often rendered inside another clickable element (e.g. the
          StackedCards front-card wrapper in DossierDetailPanel), and nested
          <button>s are invalid HTML — browsers handle focus/click on them
          inconsistently. role="button" + tabIndex + onKeyDown gets the same
          keyboard/AT semantics without the nesting problem. */}
      <span
        ref={triggerRef}
        role="button"
        tabIndex={0}
        aria-label={`Info: ${config.label}`}
        aria-expanded={isOpen}
        aria-describedby={isOpen ? popoverId : undefined}
        onMouseEnter={open}
        onMouseLeave={scheduleClose}
        onFocus={open}
        onBlur={scheduleClose}
        onClick={(event) => {
          // Mouse clicks are preceded by a native mouseenter (which already
          // called open()), so toggling here on isOpen would immediately
          // close what mouseenter just opened. Click only opens/refreshes
          // the position — closing happens via mouseleave, outside click,
          // Escape, or blur instead.
          event.stopPropagation();
          open();
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            event.stopPropagation();
            open();
          }
        }}
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "3px",
          color: theme.colors.textMuted,
          cursor: "help",
          lineHeight: 0,
          flexShrink: 0,
          ...style,
        }}
      >
        <Info size={size} aria-hidden="true" />
      </span>

      {isOpen && coords
        ? createPortal(
            <AnimatePresence>
              <motion.div
                ref={popoverRef}
                id={popoverId}
                role="tooltip"
                onMouseEnter={clearCloseTimer}
                onMouseLeave={scheduleClose}
                initial={{ opacity: 0, scale: 0.96, y: coords.placeAbove ? 4 : -4 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.96, y: coords.placeAbove ? 4 : -4 }}
                transition={{ duration: 0.12, ease: [0.4, 0, 0.2, 1] }}
                style={{
                  position: "fixed",
                  top: coords.placeAbove ? undefined : coords.top,
                  bottom: coords.placeAbove ? window.innerHeight - coords.top : undefined,
                  left: coords.left,
                  width: `${POPOVER_WIDTH}px`,
                  zIndex: 1100,
                  padding: "12px 14px",
                  borderRadius: theme.radius.md,
                  background: theme.glass.subtle.background,
                  backdropFilter: `blur(${theme.glass.subtle.blur})`,
                  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
                  border: `1px solid ${theme.glass.subtle.border}`,
                  boxShadow: "0 12px 30px rgba(0, 0, 0, 0.35)",
                  display: "flex",
                  flexDirection: "column",
                  gap: "6px",
                }}
              >
                <div style={{ color: theme.colors.textPrimary, fontSize: "0.86rem", fontWeight: 800 }}>{config.label}</div>
                <p style={{ margin: 0, color: theme.colors.textSecondary, fontSize: "0.8rem", lineHeight: 1.5 }}>
                  {config.description}
                </p>
                {config.formula ? (
                  <div
                    style={{
                      marginTop: "2px",
                      padding: "6px 8px",
                      borderRadius: theme.radius.sm,
                      background: theme.colors.panelAlt,
                      color: theme.colors.textPrimary,
                      fontSize: "0.76rem",
                      fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
                      lineHeight: 1.4,
                    }}
                  >
                    {config.formula}
                  </div>
                ) : null}
              </motion.div>
            </AnimatePresence>,
            document.body
          )
        : null}
    </>
  );
}
