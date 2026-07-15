import { useEffect, useRef, useState } from "react";
import { ChevronDown, LineChart } from "lucide-react";
import { theme } from "../ui/theme";
import MultiLayerChart, { type ChartLayer } from "./MultiLayerChart";
import TimeRangeFilter from "./TimeRangeFilter";
import PercentChangeBadge from "./PercentChangeBadge";
import { computePercentChange, filterChartLayers, normalizePriceSeries, type TimeRange } from "./chartUtils";
import { getPriceHistory, type PriceHistoryResponse } from "../../api/marketData";
import { getCompanyColor } from "../../compare/mapping";

type Props = {
  symbols: string[];
};

// EVOLVING.md EV-062, D6: wie der Einzelchart aus EV-061 startet auch der
// Kursvergleich bei 1Y.
const DEFAULT_RANGE: TimeRange = "1y";

/** Zuschaltbarer, normalisierter Kursvergleich (EVOLVING.md EV-062, D4):
 * alle Firmen als Linien, jede auf 0 % am Start ihrer (gemeinsam
 * zeitraum-geschnittenen) Serie normiert - macht Firmen mit stark
 * unterschiedlichen absoluten Kursen (z. B. 30 $ neben 800 $) direkt
 * vergleichbar. Eigenständige Komponente statt eine Mehrfirmen-Variante von
 * `PriceChartSection` (EV-061) - die beiden haben unterschiedliche
 * Lade-/Fehler-/Render-Semantik (ein Fetch vs. parallele Fetches mit
 * Teilausfall-Behandlung), eine gemeinsame Komponente mit Modus-Prop wäre
 * schwerer nachvollziehbar gewesen als zwei fokussierte Komponenten, die
 * dieselben Bausteine (TimeRangeFilter, PercentChangeBadge, MultiLayerChart,
 * chartUtils) wiederverwenden. */
export default function PriceComparisonSection({ symbols }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [range, setRange] = useState<TimeRange>(DEFAULT_RANGE);
  const [isLoading, setIsLoading] = useState(false);
  const [layers, setLayers] = useState<ChartLayer[]>([]);
  const [failedSymbols, setFailedSymbols] = useState<string[]>([]);

  const cacheRef = useRef<Map<string, PriceHistoryResponse>>(new Map());
  const loadedForRef = useRef<string | null>(null);
  const requestIdRef = useRef(0);

  function buildLayersFromCache(currentSymbols: string[], currentRange: TimeRange): ChartLayer[] {
    const rawLayers: ChartLayer[] = [];
    currentSymbols.forEach((symbol, index) => {
      const cached = cacheRef.current.get(`${symbol}:${currentRange}`);
      if (!cached) return;
      rawLayers.push({
        id: symbol,
        label: symbol,
        data: cached.rows.map((row) => ({ date: row.date, value: row.close })),
        axis: "left",
        color: getCompanyColor(index),
      });
    });

    // EV-041-Utility wiederverwendet: schneidet alle Firmen-Serien auf einen
    // GEMEINSAMEN Anker (das neueste Datum über alle Firmen), falls die
    // einzelnen Preis-Historien serverseitig leicht versetzte Enddaten
    // haben (z. B. unterschiedlich frisch gecachte Symbole).
    const synchronized = filterChartLayers(rawLayers, currentRange);
    return synchronized.map((layer) => ({ ...layer, data: normalizePriceSeries(layer.data) }));
  }

  function load(currentSymbols: string[], currentRange: TimeRange) {
    // Guard gegen veraltete Antworten (gleiches Muster wie
    // PriceChartSection#4): ein Firmenwechsel während eines laufenden
    // Fetches darf das Ergebnis eines späteren Fetches nicht überschreiben,
    // trotz Promise.allSettled.
    const requestId = ++requestIdRef.current;
    setIsLoading(true);

    const toFetch = currentSymbols.filter((symbol) => !cacheRef.current.has(`${symbol}:${currentRange}`));

    Promise.allSettled(toFetch.map((symbol) => getPriceHistory(symbol, currentRange))).then((results) => {
      const failed: string[] = [];
      results.forEach((result, i) => {
        const symbol = toFetch[i];
        if (result.status === "fulfilled") {
          cacheRef.current.set(`${symbol}:${currentRange}`, result.value);
        } else {
          failed.push(symbol);
        }
      });

      if (requestIdRef.current !== requestId) return;

      const stillMissing = currentSymbols.filter(
        (symbol) => !cacheRef.current.has(`${symbol}:${currentRange}`)
      );

      setFailedSymbols(Array.from(new Set([...failed, ...stillMissing])));
      setLayers(buildLayersFromCache(currentSymbols, currentRange));
      loadedForRef.current = `${currentSymbols.join(",")}:${currentRange}`;
      setIsLoading(false);
    });
  }

  function handleToggle() {
    const nextOpen = !isOpen;
    setIsOpen(nextOpen);
    if (nextOpen && loadedForRef.current !== `${symbols.join(",")}:${range}`) {
      load(symbols, range);
    }
  }

  function handleRangeChange(nextRange: TimeRange) {
    setRange(nextRange);
    load(symbols, nextRange);
  }

  const symbolsKey = symbols.join(",");

  // Fügt man auf der Compare-Seite eine Firma hinzu, während "Kursvergleich"
  // bereits offen ist, sollen deren Daten automatisch nachgeladen werden
  // (bisher nur handleToggle/handleRangeChange). ComparePage.tsx übergibt bei
  // jedem Render ein neues Array-Objekt - Dependency ist deshalb bewusst der
  // String symbolsKey statt des Arrays, damit der 1,5s-Poll-Tick von
  // CompareProvider nicht bei jedem Tick erneut feuert.
  useEffect(() => {
    if (isOpen && loadedForRef.current !== `${symbolsKey}:${range}`) {
      load(symbols, range);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [symbolsKey, isOpen]);

  if (symbols.length === 0) return null;

  return (
    <section style={sectionStyle}>
      <button type="button" onClick={handleToggle} style={toggleButtonStyle}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
          <LineChart size={16} color={theme.colors.chrome} />
          Kursvergleich {isOpen ? "ausblenden" : "anzeigen"}
        </span>
        <ChevronDown
          size={16}
          color={theme.colors.textMuted}
          style={{ transform: isOpen ? "rotate(180deg)" : "none", transition: `transform ${theme.motion.fast} ${theme.motion.easing}` }}
        />
      </button>

      {isOpen ? (
        <div style={{ marginTop: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px", marginBottom: "6px" }}>
            <TimeRangeFilter value={range} onChange={handleRangeChange} />
          </div>
          <div style={normalizedHintStyle}>Normalisiert auf 0 % am Zeitraumstart — keine absoluten Kurse.</div>

          {!isLoading && layers.length > 0 ? (
            <div style={percentBadgeRowStyle}>
              {symbols.map((symbol, index) => {
                const cached = cacheRef.current.get(`${symbol}:${range}`);
                if (!cached) return null;
                return (
                  <PercentChangeBadge
                    key={symbol}
                    result={computePercentChange(cached.rows.map((row) => ({ date: row.date, value: row.close })))}
                    color={getCompanyColor(index)}
                  />
                );
              })}
            </div>
          ) : null}

          {failedSymbols.length > 0 ? (
            <div style={failedChipRowStyle}>
              {failedSymbols.map((symbol) => (
                <span key={symbol} style={failedChipStyle}>
                  Kursdaten für {symbol} nicht verfügbar
                </span>
              ))}
            </div>
          ) : null}

          {isLoading ? (
            <div style={statusBoxStyle}>Kursdaten werden geladen…</div>
          ) : layers.length > 0 ? (
            <MultiLayerChart layers={layers} height={300} bucketMode="date" />
          ) : (
            <div style={statusBoxStyle}>Keine Kursdaten verfügbar.</div>
          )}
        </div>
      ) : null}
    </section>
  );
}

const sectionStyle: React.CSSProperties = {
  background: theme.colors.panelAlt,
  borderRadius: theme.radius.lg,
  padding: "18px 22px",
  border: `1px solid ${theme.colors.borderSubtle}`,
};

const toggleButtonStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  width: "100%",
  background: "none",
  border: "none",
  padding: 0,
  cursor: "pointer",
  fontSize: "0.92rem",
  fontWeight: 700,
  color: theme.colors.textPrimary,
};

const normalizedHintStyle: React.CSSProperties = {
  color: theme.colors.textMuted,
  fontSize: "0.78rem",
  marginBottom: "10px",
};

const percentBadgeRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: "8px",
  marginBottom: "10px",
};

const failedChipRowStyle: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: "8px",
  marginBottom: "10px",
};

const failedChipStyle: React.CSSProperties = {
  padding: "4px 10px",
  borderRadius: theme.radius.pill,
  background: theme.colors.dangerSoft,
  border: `1px solid ${theme.colors.dangerBorder}`,
  color: theme.colors.dangerText,
  fontSize: "0.78rem",
};

const statusBoxStyle: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  height: "160px",
  color: theme.colors.textMuted,
  fontSize: "0.92rem",
  border: `1px dashed ${theme.colors.borderSubtle}`,
  borderRadius: theme.radius.md,
};
