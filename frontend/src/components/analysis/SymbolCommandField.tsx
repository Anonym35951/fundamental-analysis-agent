import { AnimatePresence, motion } from "framer-motion";
import { Star } from "lucide-react";
import type { SymbolMeta } from "../../api/analysis";
import { theme } from "../ui/theme";
import LivePriceBadge from "../shared/LivePriceBadge";

type Props = {
  symbol: string;
  onSymbolChange: (value: string) => void;
  onFocus: () => void;
  onBlur: () => void;
  isSuggestionsOpen: boolean;
  isLoadingSymbols: boolean;
  filteredSuggestions: SymbolMeta[];
  onSelectSuggestion: (symbol: string) => void;
  isFavorited: boolean;
  onToggleFavorite: () => void;
};

export default function SymbolCommandField({
  symbol,
  onSymbolChange,
  onFocus,
  onBlur,
  isSuggestionsOpen,
  isLoadingSymbols,
  filteredSuggestions,
  onSelectSuggestion,
  isFavorited,
  onToggleFavorite,
}: Props) {
  return (
    <div style={{ position: "relative" }}>
      <label htmlFor="analysis-symbol" style={eyebrow}>
        Symbol
      </label>

      <div style={{ position: "relative", marginTop: "10px" }}>
        <input
          id="analysis-symbol"
          type="text"
          value={symbol}
          onChange={(event) => onSymbolChange(event.target.value.toUpperCase())}
          onFocus={onFocus}
          onBlur={onBlur}
          placeholder="z. B. AAPL"
          autoComplete="off"
          style={commandInput}
        />

        <button
          type="button"
          onClick={onToggleFavorite}
          disabled={!symbol}
          aria-label={isFavorited ? "Favorit entfernen" : "Als Favorit markieren"}
          title={isFavorited ? "Favorit entfernen" : "Als Favorit markieren"}
          style={{
            position: "absolute",
            top: "50%",
            right: "16px",
            transform: "translateY(-50%)",
            border: "none",
            background: "transparent",
            cursor: symbol ? "pointer" : "default",
            padding: "4px",
            display: "flex",
            alignItems: "center",
            opacity: symbol ? 1 : 0.4,
          }}
        >
          <Star size={20} color={theme.colors.chrome} fill={isFavorited ? theme.colors.chrome : "none"} />
        </button>
      </div>

      <AnimatePresence>
        {isSuggestionsOpen ? (
          <motion.div
            initial={{ opacity: 0, y: -6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.15 }}
            style={suggestionDropdown}
          >
            {isLoadingSymbols ? (
              <div style={suggestionEmptyText}>Symbole werden geladen...</div>
            ) : filteredSuggestions.length > 0 ? (
              filteredSuggestions.map((entry) => (
                <button
                  key={entry.symbol}
                  type="button"
                  onClick={() => onSelectSuggestion(entry.symbol)}
                  style={suggestionButton}
                >
                  <span style={suggestionTextColumn}>
                    <span style={suggestionSymbol}>
                      {entry.symbol}
                      {entry.name ? <span style={suggestionNameText}> — {entry.name}</span> : null}
                    </span>
                    {entry.sectors.length > 0 ? (
                      <span style={suggestionSectorText}>{entry.sectors.join(" • ")}</span>
                    ) : null}
                  </span>
                  <LivePriceBadge symbol={entry.symbol} size="sm" />
                </button>
              ))
            ) : (
              <div style={suggestionEmptyText}>Keine passenden Symbole gefunden.</div>
            )}
          </motion.div>
        ) : null}
      </AnimatePresence>
    </div>
  );
}

const eyebrow = {
  display: "block",
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const commandInput = {
  width: "100%",
  padding: "20px 56px 20px 22px",
  borderRadius: theme.radius.lg,
  border: `1px solid ${theme.colors.chromeBorder}`,
  background: theme.glass.elevated.background,
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  color: theme.colors.textPrimary,
  fontSize: "1.8rem",
  fontWeight: 800,
  letterSpacing: "0.02em",
  outline: "none",
  boxSizing: "border-box" as const,
};

const suggestionDropdown = {
  position: "absolute" as const,
  top: "100%",
  left: 0,
  right: 0,
  marginTop: "8px",
  zIndex: 20,
  display: "flex",
  flexDirection: "column" as const,
  gap: "4px",
  padding: "8px",
  maxHeight: "280px",
  overflowY: "auto" as const,
  borderRadius: theme.radius.lg,
  // Solid (non-translucent) background instead of the glass token — the
  // glass look relies on the near-black page behind it showing through a
  // blur, which doesn't give enough contrast when this dropdown opens over
  // other light-colored content (e.g. the metric picker inside a Modal).
  background: theme.colors.bgDeepAlt,
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
};

const suggestionButton = {
  display: "flex",
  flexDirection: "row" as const,
  alignItems: "center",
  justifyContent: "space-between",
  gap: "12px",
  width: "100%",
  padding: "10px 12px",
  borderRadius: theme.radius.md,
  border: "1px solid transparent",
  background: "transparent",
  color: theme.colors.textPrimary,
  cursor: "pointer",
  textAlign: "left" as const,
  transition: `background ${theme.motion.fast} ${theme.motion.easing}`,
};

const suggestionTextColumn = {
  display: "flex",
  flexDirection: "column" as const,
  alignItems: "flex-start",
  gap: "2px",
  minWidth: 0,
};

const suggestionSymbol = {
  fontSize: "0.92rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
};

const suggestionNameText = {
  fontWeight: 500,
  color: theme.colors.textSecondary,
};

const suggestionSectorText = {
  fontSize: "0.8rem",
  color: theme.colors.textMuted,
  lineHeight: 1.4,
};

const suggestionEmptyText = {
  color: theme.colors.textMuted,
  fontSize: "0.9rem",
  lineHeight: 1.6,
  padding: "8px 10px",
};
