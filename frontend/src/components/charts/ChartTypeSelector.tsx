import { useEffect, useRef, useState } from "react";
import { Check, ChevronDown } from "lucide-react";
import { theme } from "../ui/theme";
import { CHART_TYPE_LABELS, type ChartType } from "./chartUtils";

type Props = {
  value: ChartType;
  onChange: (value: ChartType) => void;
  /** Bereits über `supportedChartTypes` (chartUtils.ts) gefilterte Liste -
   * diese Komponente kennt die Eligibility-Regel selbst nicht, sie zeigt
   * nur, was der Aufrufer erlaubt. */
  options: ChartType[];
  disabled?: boolean;
};

/** Dezentes Dropdown für die Chart-Darstellung (EVOLVING.md "Erweiterbare
 * Chart-Darstellungen", CHART-004, Variante A). Reine UI-Komponente ohne
 * Datenbezug - der Wechsel selbst passiert beim Aufrufer (`onChange`), hier
 * wird nichts geladen oder berechnet. Rendert `null` bei weniger als 2
 * Optionen: ein Chart, der ohnehin nur "line" kann, braucht keinen
 * funktionslosen Button. */
export default function ChartTypeSelector({ value, onChange, options, disabled = false }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const optionRefs = useRef<Array<HTMLButtonElement | null>>([]);

  useEffect(() => {
    if (!isOpen) return;

    function handlePointerDown(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, [isOpen]);

  // Fokus auf die aktive Option, sobald das Popover aufgeht - Tastatur-
  // Bedienung landet direkt auf der aktuell gewählten Darstellung statt auf
  // der ersten Option.
  useEffect(() => {
    if (!isOpen) return;
    const activeIndex = Math.max(options.indexOf(value), 0);
    optionRefs.current[activeIndex]?.focus();
  }, [isOpen, options, value]);

  if (options.length < 2) return null;

  function select(next: ChartType) {
    setIsOpen(false);
    triggerRef.current?.focus();
    if (next !== value) onChange(next);
  }

  function handleTriggerKeyDown(event: React.KeyboardEvent) {
    if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      setIsOpen(true);
    } else if (event.key === "Escape") {
      setIsOpen(false);
    }
  }

  function handleOptionKeyDown(event: React.KeyboardEvent, index: number) {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      optionRefs.current[(index + 1) % options.length]?.focus();
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      optionRefs.current[(index - 1 + options.length) % options.length]?.focus();
    } else if (event.key === "Escape") {
      event.preventDefault();
      setIsOpen(false);
      triggerRef.current?.focus();
    } else if (event.key === "Tab") {
      setIsOpen(false);
    }
  }

  return (
    <div ref={containerRef} style={{ position: "relative", display: "inline-block" }}>
      <button
        ref={triggerRef}
        type="button"
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-label="Chart-Darstellung ändern"
        onClick={() => setIsOpen((prev) => !prev)}
        onKeyDown={handleTriggerKeyDown}
        style={triggerStyle(disabled)}
      >
        <span>
          Darstellung: <strong>{CHART_TYPE_LABELS[value]}</strong>
        </span>
        <ChevronDown
          size={14}
          style={{
            transform: isOpen ? "rotate(180deg)" : "none",
            transition: `transform ${theme.motion.fast} ${theme.motion.easing}`,
            flexShrink: 0,
          }}
        />
      </button>

      {isOpen ? (
        <div role="listbox" aria-label="Darstellung" style={popoverStyle}>
          {options.map((option, index) => {
            const isSelected = option === value;
            return (
              <button
                key={option}
                ref={(el) => {
                  optionRefs.current[index] = el;
                }}
                type="button"
                role="option"
                aria-selected={isSelected}
                onClick={() => select(option)}
                onKeyDown={(event) => handleOptionKeyDown(event, index)}
                style={optionStyle(isSelected)}
              >
                <Check size={14} style={{ opacity: isSelected ? 1 : 0, flexShrink: 0 }} />
                {CHART_TYPE_LABELS[option]}
              </button>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function triggerStyle(disabled: boolean): React.CSSProperties {
  return {
    display: "inline-flex",
    alignItems: "center",
    gap: "8px",
    minHeight: "40px",
    padding: "8px 14px",
    borderRadius: theme.radius.pill,
    border: `1px solid ${theme.glass.subtle.border}`,
    background: theme.glass.subtle.background,
    color: theme.colors.textSecondary,
    fontSize: "0.82rem",
    fontWeight: 600,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.55 : 1,
    whiteSpace: "nowrap",
    transition: `border-color ${theme.motion.fast} ${theme.motion.easing}`,
  };
}

const popoverStyle: React.CSSProperties = {
  position: "absolute",
  top: "calc(100% + 6px)",
  right: 0,
  zIndex: 30,
  display: "flex",
  flexDirection: "column",
  gap: "2px",
  minWidth: "140px",
  padding: "6px",
  borderRadius: theme.radius.md,
  background: theme.colors.bgDeepAlt,
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
};

function optionStyle(isSelected: boolean): React.CSSProperties {
  return {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    width: "100%",
    minHeight: "36px",
    padding: "6px 10px",
    borderRadius: theme.radius.sm,
    border: "1px solid transparent",
    background: isSelected ? theme.colors.chromeSoft : "transparent",
    color: theme.colors.textPrimary,
    fontSize: "0.86rem",
    fontWeight: isSelected ? 700 : 500,
    cursor: "pointer",
    textAlign: "left",
  };
}
