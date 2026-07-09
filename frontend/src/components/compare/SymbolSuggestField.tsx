import { useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Input from "../ui/Input";
import { theme } from "../ui/theme";
import { useSymbolSearch } from "../../hooks/useSymbolSearch";
import LivePriceBadge from "../shared/LivePriceBadge";

type Props = {
  value: string;
  onChange: (value: string) => void;
  onCommit: (value: string) => void;
  placeholder?: string;
};

/** Compact symbol input for the Compare page's company rows — opens a
 * server-seitig durchsuchtes Dropdown on focus/click (debounced query-as-
 * you-type, siehe useSymbolSearch). Mirrors the blur-timeout pattern from
 * SymbolCommandField / AnalyzePage so a suggestion click registers before
 * the input's blur would otherwise close the dropdown first. */
export default function SymbolSuggestField({ value, onChange, onCommit, placeholder }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const blurTimeoutRef = useRef<number | null>(null);

  const normalized = value.trim().toUpperCase();
  const { suggestions: filteredSuggestions, isLoadingSuggestions: isLoadingSymbols } = useSymbolSearch(normalized);

  function handleFocus() {
    if (blurTimeoutRef.current) {
      window.clearTimeout(blurTimeoutRef.current);
    }
    setIsOpen(true);
  }

  function handleBlur() {
    blurTimeoutRef.current = window.setTimeout(() => {
      setIsOpen(false);
      onCommit(value);
    }, 120);
  }

  function handleSelect(symbol: string) {
    if (blurTimeoutRef.current) {
      window.clearTimeout(blurTimeoutRef.current);
    }
    setIsOpen(false);
    onChange(symbol);
    onCommit(symbol);
  }

  return (
    <div style={{ position: "relative", flex: 1 }}>
      <Input
        value={value}
        onChange={(e) => onChange(e.target.value.toUpperCase())}
        onFocus={handleFocus}
        onBlur={handleBlur}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            if (blurTimeoutRef.current) window.clearTimeout(blurTimeoutRef.current);
            setIsOpen(false);
            onCommit(value);
          }
        }}
        placeholder={placeholder}
        autoComplete="off"
      />

      <AnimatePresence>
        {isOpen ? (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.12 }}
            style={dropdownStyle}
          >
            {isLoadingSymbols ? (
              <div style={emptyTextStyle}>Symbole werden geladen...</div>
            ) : filteredSuggestions.length > 0 ? (
              filteredSuggestions.map((entry) => (
                <button key={entry.symbol} type="button" onClick={() => handleSelect(entry.symbol)} style={suggestionButtonStyle}>
                  <span style={suggestionTextColumnStyle}>
                    <span style={suggestionSymbolStyle}>
                      {entry.symbol}
                      {entry.name ? <span style={suggestionNameStyle}> — {entry.name}</span> : null}
                    </span>
                    {entry.sectors.length > 0 ? (
                      <span style={suggestionSectorStyle}>{entry.sectors.join(" • ")}</span>
                    ) : null}
                  </span>
                  <LivePriceBadge symbol={entry.symbol} size="sm" />
                </button>
              ))
            ) : (
              <div style={emptyTextStyle}>Keine passenden Symbole gefunden.</div>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

const dropdownStyle: React.CSSProperties = {
  position: "absolute",
  top: "100%",
  left: 0,
  right: 0,
  marginTop: "6px",
  zIndex: 30,
  display: "flex",
  flexDirection: "column",
  gap: "2px",
  padding: "6px",
  maxHeight: "220px",
  overflowY: "auto",
  borderRadius: theme.radius.md,
  background: theme.colors.bgDeepAlt,
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
};

const suggestionButtonStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "10px",
  width: "100%",
  padding: "8px 10px",
  borderRadius: theme.radius.sm,
  border: "1px solid transparent",
  background: "transparent",
  color: theme.colors.textPrimary,
  cursor: "pointer",
  textAlign: "left",
};

const suggestionTextColumnStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "flex-start",
  gap: "2px",
  minWidth: 0,
};

const suggestionSymbolStyle: React.CSSProperties = {
  fontSize: "0.88rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
};

const suggestionSectorStyle: React.CSSProperties = {
  fontSize: "0.78rem",
  color: theme.colors.textMuted,
};

const suggestionNameStyle: React.CSSProperties = {
  fontWeight: 500,
  color: theme.colors.textSecondary,
};

const emptyTextStyle: React.CSSProperties = {
  color: theme.colors.textMuted,
  fontSize: "0.85rem",
  padding: "8px 10px",
};
