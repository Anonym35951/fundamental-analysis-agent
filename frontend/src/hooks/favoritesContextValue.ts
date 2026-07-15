import { createContext } from "react";
import type { FavoriteEntry } from "../api/favorites";

export type FavoritesContextValue = {
  favorites: FavoriteEntry[];
  isLoading: boolean;
  isFavorite: (symbol: string) => boolean;
  toggleFavorite: (symbol: string) => Promise<void>;
};

// Eigene Datei ohne Komponenten-Export (react-refresh/only-export-components,
// LAUNCH_AUDIT.md P2-10) - spiegelt analysisJobsContextValue.ts/compareContextValue.ts.
export const FavoritesContext = createContext<FavoritesContextValue | null>(null);
