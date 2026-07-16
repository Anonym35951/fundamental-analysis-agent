import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { addFavorite, getFavorites, removeFavorite, type FavoriteEntry } from "../api/favorites";
import { FavoritesContext, type FavoritesContextValue } from "./favoritesContextValue";

/** Globaler Favoriten-Store, oberhalb des Routers gemountet (wie
 * CompareProvider/AnalysisJobsProvider) - ein einziger Fetch pro Session statt
 * je einem unabhängigen `getFavorites()`-Aufruf in Sidebar, AnalyzePage und
 * Dashboard. Dadurch sieht jeder Verbraucher einen Favoriten-Toggle sofort,
 * ohne Reload (KORREKTUREN.md Punkt 4). */
export function FavoritesProvider({ children }: { children: ReactNode }) {
  const [favorites, setFavorites] = useState<FavoriteEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  const fetchFavorites = useCallback(() => {
    // Bug-Fix (2026-07-16): FavoritesProvider ist oberhalb des Routers
    // gemountet und lief bisher IMMER an, auch fuer ausgeloggte Besucher auf
    // Landing/Login/Register. /favorites verlangt Auth (Depends(get_current_
    // user), FastAPIs OAuth2PasswordBearer wirft ohne Header sofort 401) -
    // jeder abgemeldete Aufruf endete in einem 401, das api/client.ts's
    // clearSessionAndRedirectToLogin() mit einem HARTEN
    // `window.location.href = "/login"` beantwortet. Sichtbar als: Seite
    // rendert (Header/Footer aus dem statischen Layout erscheinen), dann
    // reisst der Hard-Redirect alles weg und /login laedt neu - u. a. auch
    // beim direkten Aufruf von /register. Ohne Token also gar nicht erst
    // fetchen; app:login (nach echtem Login) fetcht ohnehin nach.
    if (!localStorage.getItem("access_token")) {
      setFavorites([]);
      setIsLoading(false);
      return;
    }
    setIsLoading(true);
    getFavorites()
      .then(setFavorites)
      .catch(() => setFavorites([]))
      .finally(() => setIsLoading(false));
  }, []);

  useEffect(() => {
    fetchFavorites();
  }, [fetchFavorites]);

  // FavoritesProvider fetcht nur einmalig beim App-Mount (oberhalb des
  // Routers). LoginPage.tsx loggt per Client-Navigation ein (kein Reload) -
  // lief der Erst-Fetch vor dem Login (401 -> favorites: []), bliebe die
  // Liste sonst für die ganze Session leer. Fix: erneut fetchen, sobald
  // LoginPage das app:login-Event dispatcht.
  useEffect(() => {
    window.addEventListener("app:login", fetchFavorites);
    return () => window.removeEventListener("app:login", fetchFavorites);
  }, [fetchFavorites]);

  // Gleiche Logout-Konvention wie CompareProvider/AnalyzeWorkspaceProvider:
  // api/client.ts dispatcht dieses Event von jedem Logout-Pfad aus, damit die
  // Favoritenliste eines Nutzers nicht in der Session des nächsten sichtbar
  // bleibt.
  useEffect(() => {
    const clear = () => setFavorites([]);
    window.addEventListener("app:logout", clear);
    return () => window.removeEventListener("app:logout", clear);
  }, []);

  const isFavorite = useCallback(
    (symbol: string) => favorites.some((fav) => fav.symbol === symbol),
    [favorites]
  );

  const toggleFavorite = useCallback(async (symbol: string) => {
    const cleanSymbol = symbol.trim().toUpperCase();
    if (!cleanSymbol) return;

    // wasFavorite/removedEntry werden synchron innerhalb des funktionalen
    // Updaters berechnet (React führt ihn synchron gegen den zuletzt
    // eingereihten State aus) - unabhängig vom Timing des äußeren Renders,
    // damit zwei Toggles verschiedener Symbole im selben Tick sich nicht
    // gegenseitig überschreiben (stale `favorites`-Closure).
    let wasFavorite = false;
    let removedEntry: FavoriteEntry | undefined;
    setFavorites((current) => {
      wasFavorite = current.some((fav) => fav.symbol === cleanSymbol);
      if (wasFavorite) {
        removedEntry = current.find((fav) => fav.symbol === cleanSymbol);
        return current.filter((fav) => fav.symbol !== cleanSymbol);
      }
      return [...current, { symbol: cleanSymbol, created_at: new Date().toISOString() }];
    });

    try {
      if (wasFavorite) {
        await removeFavorite(cleanSymbol);
      } else {
        await addFavorite(cleanSymbol);
      }
    } catch (error) {
      // Relativer statt absoluter Rollback: macht nur DIESEN einen Toggle
      // rückgängig, statt einen Snapshot zurückzuschreiben, der einen
      // zwischenzeitlich erfolgreichen zweiten Toggle mit wegwischen würde.
      setFavorites((current) =>
        wasFavorite
          ? [...current, removedEntry ?? { symbol: cleanSymbol, created_at: new Date().toISOString() }]
          : current.filter((fav) => fav.symbol !== cleanSymbol)
      );
      throw error;
    }
  }, []);

  const value = useMemo<FavoritesContextValue>(
    () => ({ favorites, isLoading, isFavorite, toggleFavorite }),
    [favorites, isLoading, isFavorite, toggleFavorite]
  );

  return <FavoritesContext.Provider value={value}>{children}</FavoritesContext.Provider>;
}
