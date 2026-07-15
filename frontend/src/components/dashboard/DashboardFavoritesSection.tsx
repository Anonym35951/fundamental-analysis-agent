import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { theme } from "../ui/theme";
import LivePriceBadge from "../shared/LivePriceBadge";
import Sparkline from "../charts/Sparkline";
import PercentChangeBadge from "../charts/PercentChangeBadge";
import { computePercentChange } from "../charts/chartUtils";
import { getFavorites, type FavoriteEntry } from "../../api/favorites";
import { getPriceHistoryBatch, type PriceHistoryBatchEntry } from "../../api/marketData";

// EVOLVING.md EV-071: erste 10 Favoriten in einem Batch-Call, Rest erst
// nach "Mehr anzeigen" (Payload-/yfinance-Schonung, analog EV-070).
const INITIAL_VISIBLE_COUNT = 10;
const SPARKLINE_RANGE = "1m" as const;

type BatchState = Record<string, PriceHistoryBatchEntry>;

/** Dashboard-Sektion "Favoriten" (EVOLVING.md EV-071): Symbol, Live-Kurs,
 * 1M-Sparkline, 1M-%-Veränderung. Ergänzt die bestehende Sidebar-
 * Favoritenliste, ersetzt sie nicht - beide bleiben unabhängig
 * nebeneinander bestehen. */
export default function DashboardFavoritesSection() {
  const [favorites, setFavorites] = useState<FavoriteEntry[] | null>(null);
  const [batchData, setBatchData] = useState<BatchState>({});
  const [batchError, setBatchError] = useState(false);
  const [isLoadingBatch, setIsLoadingBatch] = useState(false);
  const [visibleCount, setVisibleCount] = useState(INITIAL_VISIBLE_COUNT);

  useEffect(() => {
    let isMounted = true;
    getFavorites()
      .then((data) => {
        if (isMounted) setFavorites(data);
      })
      .catch(() => {
        if (isMounted) setFavorites([]);
      });
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    if (!favorites || favorites.length === 0) return;

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
    if (!favorites) return;
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

  if (favorites === null) {
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
        {visibleFavorites.map((favorite) => {
          const entry = batchData[favorite.symbol];
          const rows = entry && "rows" in entry ? entry.rows : null;
          const sparklineData = rows?.map((row) => ({ date: row.date, value: row.close })) ?? [];
          const percentResult = rows && rows.length > 0
            ? computePercentChange(rows.map((row) => ({ date: row.date, value: row.close })))
            : null;

          return (
            <div key={favorite.symbol} style={favoriteCard}>
              <div style={favoriteCardHeader}>
                <span style={favoriteSymbol}>{favorite.symbol}</span>
                <LivePriceBadge symbol={favorite.symbol} size="sm" />
              </div>

              {isLoadingBatch && !entry ? (
                <div style={sparklinePlaceholder}>Lädt…</div>
              ) : sparklineData.length >= 2 ? (
                <Sparkline data={sparklineData} color={theme.colors.chrome} height={40} />
              ) : (
                <div style={sparklinePlaceholder}>–</div>
              )}

              <div style={favoriteCardFooter}>
                {percentResult ? (
                  <PercentChangeBadge result={percentResult} />
                ) : (
                  <span style={dashText}>–</span>
                )}
              </div>
            </div>
          );
        })}
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
  justifyContent: "flex-end",
};

const sparklinePlaceholder: React.CSSProperties = {
  height: "40px",
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
