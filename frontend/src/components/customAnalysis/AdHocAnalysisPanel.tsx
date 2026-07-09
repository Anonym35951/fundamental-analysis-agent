import { useState } from "react";
import { Link } from "react-router-dom";
import type { SymbolMeta } from "../../api/analysis";
import type { MetricCatalogEntry, MetricSelection } from "../../types/customAnalysis";
import { Button } from "../ui";
import { theme } from "../ui/theme";
import SymbolCommandField from "../analysis/SymbolCommandField";
import MetricCatalogPicker from "./MetricCatalogPicker";

type Props = {
  catalog: MetricCatalogEntry[];
  isLoadingCatalog: boolean;

  symbol: string;
  onSymbolChange: (value: string) => void;
  onSymbolFocus: () => void;
  onSymbolBlur: () => void;
  isSuggestionsOpen: boolean;
  isLoadingSymbols: boolean;
  filteredSuggestions: SymbolMeta[];
  onSelectSuggestion: (symbol: string) => void;
  isFavorited: boolean;
  onToggleFavorite: () => void;

  isStarting: boolean;
  isLimitReached: boolean;
  onStart: (metrics: MetricSelection[]) => void;
};

/** Symbol + metric picker + immediate "Analyse starten" in one overlay, so a
 * one-off lookup doesn't require closing the modal and using the page-level
 * symbol field separately. Nothing here is persisted as a definition. */
export default function AdHocAnalysisPanel({
  catalog,
  isLoadingCatalog,
  symbol,
  onSymbolChange,
  onSymbolFocus,
  onSymbolBlur,
  isSuggestionsOpen,
  isLoadingSymbols,
  filteredSuggestions,
  onSelectSuggestion,
  isFavorited,
  onToggleFavorite,
  isStarting,
  isLimitReached,
  onStart,
}: Props) {
  const [metrics, setMetrics] = useState<MetricSelection[]>([]);

  const canStart = Boolean(symbol.trim()) && metrics.length > 0 && !isStarting && !isLimitReached;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "18px" }}>
      <SymbolCommandField
        symbol={symbol}
        onSymbolChange={onSymbolChange}
        onFocus={onSymbolFocus}
        onBlur={onSymbolBlur}
        isSuggestionsOpen={isSuggestionsOpen}
        isLoadingSymbols={isLoadingSymbols}
        filteredSuggestions={filteredSuggestions}
        onSelectSuggestion={onSelectSuggestion}
        isFavorited={isFavorited}
        onToggleFavorite={onToggleFavorite}
      />

      <MetricCatalogPicker catalog={catalog} isLoadingCatalog={isLoadingCatalog} onChange={setMetrics} />

      {isLimitReached ? (
        <p style={{ margin: 0, color: theme.colors.dangerText, fontSize: "0.92rem", lineHeight: 1.6 }}>
          Monatliches Free-Plan-Kontingent aufgebraucht. Neue Analysen gibt's
          zum Monatsanfang — oder{" "}
          <Link to="/app/billing" style={{ color: theme.colors.dangerText, fontWeight: 700 }}>
            jetzt zu Pro wechseln
          </Link>{" "}
          für unbegrenzte Analysen.
        </p>
      ) : null}

      <div style={{ display: "flex", justifyContent: "flex-end" }}>
        <Button
          variant="cta"
          disabled={!canStart}
          onClick={() => onStart(metrics)}
          style={{ padding: "12px 22px" }}
        >
          {isStarting ? "Analyse wird gestartet..." : "Analyse starten"}
        </Button>
      </div>
    </div>
  );
}
