import { apiRequest } from "./client";

export type FavoriteEntry = {
  symbol: string;
  created_at: string;
};

export async function getFavorites(): Promise<FavoriteEntry[]> {
  return apiRequest<FavoriteEntry[]>("/favorites");
}

export async function addFavorite(symbol: string): Promise<FavoriteEntry> {
  return apiRequest<FavoriteEntry>(
    `/favorites?symbol=${encodeURIComponent(symbol)}`,
    { method: "POST" }
  );
}

export async function removeFavorite(symbol: string): Promise<void> {
  return apiRequest<void>(`/favorites/${encodeURIComponent(symbol)}`, {
    method: "DELETE",
  });
}
