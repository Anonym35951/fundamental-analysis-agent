import { useRef, useState } from "react";
import { ChevronDown, LineChart } from "lucide-react";
import { theme } from "../ui/theme";
import MultiLayerChart from "./MultiLayerChart";
import TimeRangeFilter from "./TimeRangeFilter";
import PercentChangeBadge from "./PercentChangeBadge";
import { computePercentChange, type TimeRange } from "./chartUtils";
import { getPriceHistory, type PriceHistoryResponse } from "../../api/marketData";

type Props = {
  symbol: string;
};

// EVOLVING.md EV-061, D6: Kurscharts starten bei 1Y (anders als
// Fundamental-Charts, die bei "max" starten - tägliche Kursdaten sind auch
// im 1-Jahres-Fenster gut lesbar, Fundamentaldaten dagegen haben dort oft
// nur 1 Datenpunkt).
const DEFAULT_RANGE: TimeRange = "1y";

/** Zuschaltbarer Kurschart (EVOLVING.md EV-061) - lädt bewusst NICHTS, bevor
 * der Nutzer den Abschnitt öffnet (kein API-Call ohne Interaktion). Range-
 * Wechsel filtert serverseitig (EV-060) statt clientseitig wie bei den
 * Fundamental-Charts (EV-040/041), da die Rohserie hier täglich und über
 * Jahrzehnte potenziell groß ist. Einfacher In-Memory-Cache pro Symbol+Range
 * im Component-State, damit ein Zurückwechseln zu einer bereits geladenen
 * Range keinen erneuten Request auslöst. */
export default function PriceChartSection({ symbol }: Props) {
  const [isOpen, setIsOpen] = useState(false);
  const [range, setRange] = useState<TimeRange>(DEFAULT_RANGE);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<PriceHistoryResponse | null>(null);

  const cacheRef = useRef<Map<string, PriceHistoryResponse>>(new Map());
  const loadedForRef = useRef<string | null>(null);

  function load(nextSymbol: string, nextRange: TimeRange) {
    const cacheKey = `${nextSymbol}:${nextRange}`;
    const cached = cacheRef.current.get(cacheKey);
    if (cached) {
      setData(cached);
      setError(null);
      loadedForRef.current = cacheKey;
      return;
    }

    setIsLoading(true);
    setError(null);
    getPriceHistory(nextSymbol, nextRange)
      .then((response) => {
        cacheRef.current.set(cacheKey, response);
        loadedForRef.current = cacheKey;
        setData(response);
      })
      .catch(() => {
        setError("Kursdaten konnten nicht geladen werden.");
      })
      .finally(() => setIsLoading(false));
  }

  function handleToggle() {
    const nextOpen = !isOpen;
    setIsOpen(nextOpen);
    if (nextOpen && loadedForRef.current !== `${symbol}:${range}`) {
      load(symbol, range);
    }
  }

  function handleRangeChange(nextRange: TimeRange) {
    setRange(nextRange);
    load(symbol, nextRange);
  }

  const layer = data
    ? {
        id: symbol,
        label: symbol,
        data: data.rows.map((row) => ({ date: row.date, value: row.close })),
        axis: "left" as const,
        color: theme.colors.chrome,
        currency: data.currency,
      }
    : null;

  return (
    <section style={sectionStyle}>
      <button type="button" onClick={handleToggle} style={toggleButtonStyle}>
        <span style={{ display: "inline-flex", alignItems: "center", gap: "8px" }}>
          <LineChart size={16} color={theme.colors.chrome} />
          Kursentwicklung {isOpen ? "ausblenden" : "anzeigen"}
        </span>
        <ChevronDown
          size={16}
          color={theme.colors.textMuted}
          style={{ transform: isOpen ? "rotate(180deg)" : "none", transition: `transform ${theme.motion.fast} ${theme.motion.easing}` }}
        />
      </button>

      {isOpen ? (
        <div style={{ marginTop: "16px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: "12px", marginBottom: "14px" }}>
            <TimeRangeFilter value={range} onChange={handleRangeChange} />
            {data && data.rows.length > 0 ? (
              <PercentChangeBadge result={computePercentChange(data.rows.map((row) => ({ date: row.date, value: row.close })))} />
            ) : null}
          </div>

          {isLoading ? (
            <div style={statusBoxStyle}>Kursdaten werden geladen…</div>
          ) : error ? (
            <div style={statusBoxStyle}>{error}</div>
          ) : layer && layer.data.length > 0 ? (
            <MultiLayerChart layers={[layer]} height={280} bucketMode="date" />
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
