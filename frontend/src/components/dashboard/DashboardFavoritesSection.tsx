import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { theme } from "../ui/theme";
import LivePriceBadge from "../shared/LivePriceBadge";
import Sparkline from "../charts/Sparkline";
import PercentChangeBadge from "../charts/PercentChangeBadge";
import { appendLivePoint, computePercentChange, filterToLastDays, localIsoDate } from "../charts/chartUtils";
import { useFavorites } from "../../hooks/useFavoritesContext";
import { useLivePrice } from "../../hooks/useLivePrice";
import { getPriceHistoryBatch, type PriceHistoryBatchEntry } from "../../api/marketData";

// EVOLVING.md EV-071: erste 10 Favoriten in einem Batch-Call, Rest erst
// nach "Mehr anzeigen" (Payload-/yfinance-Schonung, analog EV-070).
const INITIAL_VISIBLE_COUNT = 10;
// EVOLVING.md CH-007: der Batch-Endpoint kennt serverseitig nur "1m"/"3m" —
// "1m" wird weiterhin geladen (genug Handelstage, um daraus verlässlich eine
// Woche herauszuschneiden, auch über Feiertage hinweg), aber NUR die letzten
// FAVORITES_DISPLAY_WINDOW_DAYS Kalendertage werden angezeigt/berechnet
// (Chart + %-Änderung) — kein zusätzlicher Request nötig.
const SPARKLINE_RANGE = "1m" as const;
const FAVORITES_DISPLAY_WINDOW_DAYS = 7;
const FAVORITES_DISPLAY_WINDOW_LABEL = "1W";

type BatchState = Record<string, PriceHistoryBatchEntry>;

/** Dashboard-Sektion "Favoriten" (EVOLVING.md EV-071): Symbol, Live-Kurs,
 * 1W-Sparkline, 1W-%-Veränderung. Ergänzt die bestehende Sidebar-
 * Favoritenliste, ersetzt sie nicht - beide bleiben unabhängig
 * nebeneinander bestehen. */
export default function DashboardFavoritesSection() {
  const { favorites, isLoading } = useFavorites();
  const [batchData, setBatchData] = useState<BatchState>({});
  const [batchError, setBatchError] = useState(false);
  const [isLoadingBatch, setIsLoadingBatch] = useState(false);
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_COUNT);

  useEffect(() => {
    if (favorites.length === 0) return;

    const symbols = favorites.slice(0, INITIAL_VISIBLE_COUNT).map((f) => f.symbol);
    const alreadyLoaded = symbols.every((symbol) => symbol in batchData);
    if (alreadyLoaded) return;

    let isMounted = true;
    setIsLoadingBatch(true);
    getPriceHistoryBatch(symbols, SPARKLINE_RANGE)
      .then((response) => {
        if (!isMounted) return;
        const next: BatchState = {};
        for (const entry of response.results) next[entry.symbol] = entry;
        setBatchData((prev) => ({ ...prev, ...next }));
      })
      .catch(() => {
        if (isMounted) setBatchError(true);
      })
      .finally(() => {
        if (isMounted) setIsLoadingBatch(false);
      });

    return () => {
      isMounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [favorites]);

  function handleShowMore() {
    const nextVisible = favorites.length;
    const symbols = favorites.slice(visibleCount, nextVisible).map((f) => f.symbol);
    setVisibleCount(nextVisible);

    const alreadyLoaded = symbols.every((symbol) => symbol in batchData);
    if (alreadyLoaded || symbols.length === 0) return;

    setIsLoadingBatch(true);
    getPriceHistoryBatch(symbols, SPARKLINE_RANGE)
      .then((response) => {
        const next: BatchState = {};
        for (const entry of response.results) next[entry.symbol] = entry;
        setBatchData((prev) => ({ ...prev, ...next }));
      })
      .catch(() => setBatchError(true))
      .finally(() => setIsLoadingBatch(false));
  }

  if (isLoading) {
    return (
      <div style={panel}>
        <div style={panelTitle}>Favoriten</div>
        <div style={emptyState}>Wird geladen...</div>
      </div>
    );
  }

  if (favorites.length === 0) {
    return (
      <div style={panel}>
        <div style={panelTitle}>Favoriten</div>
        <div style={emptyState}>
          Noch keine Favoriten markiert. Markiere ein Unternehmen auf der{" "}
          <Link to="/app/analyze" style={{ color: theme.colors.chrome, fontWeight: 700 }}>
            Analyseseite
          </Link>{" "}
          als Favorit, um es hier mit Kursverlauf zu sehen.
        </div>
      </div>
    );
  }

  const visibleFavorites = favorites.slice(0, visibleCount);

  return (
    <div style={panel}>
      <div style={panelTitle}>Favoriten</div>

      <div style={cardGrid}>
        {visibleFavorites.map((favorite) => (
          <FavoriteCard
            key={favorite.symbol}
            symbol={favorite.symbol}
            entry={batchData[favorite.symbol]}
            isLoadingBatch={isLoadingBatch}
          />
        ))}
      </div>

      {batchError ? (
        <div style={batchErrorHint}>
          Kursverläufe konnten nicht vollständig geladen werden — Kurse werden trotzdem angezeigt.
        </div>
      ) : null}

      {visibleCount < favorites.length ? (
        <button type="button" onClick={handleShowMore} style={showMoreButton}>
          {favorites.length - visibleCount} weitere Favoriten anzeigen
        </button>
      ) : null}
    </div>
  );
}

type FavoriteCardProps = {
  symbol: string;
  entry: PriceHistoryBatchEntry | undefined;
  isLoadingBatch: boolean;
};

/** Eine Favoriten-Karte (EVOLVING.md CH-005). Eigene Komponente statt inline
 * im .map(), weil der useLivePrice-Hook nicht in einer Schleife aufrufbar
 * ist. Der Hook erzeugt KEINE zusätzlichen Requests: er wird zweiter
 * Listener desselben priceStore-Eintrags, den das LivePriceBadge daneben
 * ohnehin schon abonniert (geteilter 20s-Poll, in-flight-Dedupe). */
function FavoriteCard({ symbol, entry, isLoadingBatch }: FavoriteCardProps) {
  const { price } = useLivePrice(symbol);

  const rows = entry && "rows" in entry ? entry.rows : null;
  const baseSeries = rows?.map((row) => ({ date: row.date, value: row.close })) ?? [];
  // CH-007: erst auf die letzte Woche der ECHTEN Historie zuschneiden (Anker
  // = spätestes Datum in baseSeries selbst), DANN den Live-Preis anhängen —
  // nicht umgekehrt. Grund: der Preis-History-Cache kann gegenüber dem
  // aktuellen Kalendertag hinterherhinken (Wochenende/Feiertag/Cache-TTL);
  // würde zuerst der auf "heute" datierte Live-Punkt angehängt und danach
  // auf 7 Tage zurückgeschnitten, könnte genau dieser Rückstand die
  // gesamte echte Historie aus dem Fenster herausfiltern und nur den
  // einzelnen Live-Punkt übriglassen (leerer Chart, "n. v."-Badge).
  const lastWeek = filterToLastDays(baseSeries, FAVORITES_DISPLAY_WINDOW_DAYS);
  // CH-005: Live-Preis als letzter Chartpunkt, damit das Chart-Ende dem
  // daneben angezeigten Live-Preis entspricht (beobachtete PYPL-Diskrepanz:
  // die History endet am letzten Tages-Close, das Badge pollt intraday).
  // Wird NACH dem Zuschnitt angehängt, damit er nie herausgefiltert wird.
  const sparklineData = appendLivePoint(lastWeek, price, localIsoDate());
  // %-Badge rechnet auf derselben Serie — Badge und Chart-Ende bleiben
  // konsistent (beide inkl. Live-Preis, beide 1 Woche).
  const percentResult = sparklineData.length > 0 ? computePercentChange(sparklineData) : null;

  return (
    <div style={favoriteCard}>
      <div style={favoriteCardHeader}>
        <span style={favoriteSymbol}>{symbol}</span>
        <LivePriceBadge symbol={symbol} size="sm" />
      </div>

      {isLoadingBatch && !entry ? (
        <div style={sparklinePlaceholder}>Lädt…</div>
      ) : sparklineData.length >= 2 ? (
        <Sparkline
          data={sparklineData}
          color={theme.colors.chrome}
          height={48}
          showStartEndAxis
          currency={entry && "rows" in entry ? entry.currency : null}
        />
      ) : (
        <div style={sparklinePlaceholder}>–</div>
      )}

      <div style={favoriteCardFooter}>
        {entry && "rows" in entry ? (
          <span style={rangeLabelText}>{FAVORITES_DISPLAY_WINDOW_LABEL}</span>
        ) : (
          <span />
        )}
        {percentResult ? (
          <PercentChangeBadge result={percentResult} />
        ) : (
          <span style={dashText}>–</span>
        )}
      </div>
    </div>
  );
}

const panel: React.CSSProperties = {
  background: theme.glass.subtle.background,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
  borderRadius: theme.radius.lg,
  padding: "28px",
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
  minHeight: "170px",
  boxShadow: "0 14px 34px rgba(0, 0, 0, 0.22)",
  boxSizing: "border-box",
};

const panelTitle: React.CSSProperties = {
  fontSize: "1.42rem",
  fontWeight: 800,
  marginBottom: "18px",
  color: theme.colors.textPrimary,
};

const emptyState: React.CSSProperties = {
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  lineHeight: 1.8,
};

const cardGrid: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(160px, 1fr))",
  gap: "12px",
};

const favoriteCard: React.CSSProperties = {
  padding: "14px",
  borderRadius: theme.radius.md,
  background: theme.colors.panel,
  border: `1px solid ${theme.colors.borderSubtle}`,
  display: "flex",
  flexDirection: "column",
  gap: "8px",
};

const favoriteCardHeader: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "8px",
};

const favoriteSymbol: React.CSSProperties = {
  fontWeight: 800,
  fontSize: "0.98rem",
  color: theme.colors.textPrimary,
};

const favoriteCardFooter: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "8px",
};

// CH-005: TimeFrame-Herkunft der Chart-Daten ("1M") — Quelle ist das bisher
// ungenutzte range-Feld der price-history-Response, kein Umschalter
// (Produktentscheidung; Batch-Endpoint kann serverseitig nur 1m/3m).
const rangeLabelText: React.CSSProperties = {
  color: theme.colors.textMuted,
  fontSize: "0.74rem",
  fontWeight: 700,
  letterSpacing: "0.04em",
};

const sparklinePlaceholder: React.CSSProperties = {
  height: "48px",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: theme.colors.textMuted,
  fontSize: "0.82rem",
};

const dashText: React.CSSProperties = {
  color: theme.colors.textMuted,
  fontSize: "0.82rem",
};

const batchErrorHint: React.CSSProperties = {
  marginTop: "12px",
  color: theme.colors.textMuted,
  fontSize: "0.82rem",
};

const showMoreButton: React.CSSProperties = {
  marginTop: "14px",
  display: "inline-flex",
  alignItems: "center",
  gap: "8px",
  background: "none",
  border: "none",
  color: theme.colors.chrome,
  fontWeight: 700,
  fontSize: "0.9rem",
  cursor: "pointer",
  padding: 0,
};
