import { useCallback } from "react";
import { useLocale } from "./useLocale";
import type { Dictionary, Namespace } from "./locales/de";
import { resolvePath, type DotPaths } from "./types";
import { interpolate } from "./t";

/** `t()`-Aufrufe sind gegen den DE-Schema-Namespace typisiert (§ 13) — ein
 * nicht existierender Key ist ein TS-Fehler. Fallback-Kette zur Laufzeit
 * (§ 15): aktive Locale → DE-Fallback → Dev-Warning + Roh-Key. Produktion
 * zeigt dank Compile-Zeit-Typisierung praktisch nie einen rohen Key. */
export function useTranslation<NS extends Namespace>(ns: NS) {
  const { locale, dictionary, fallbackDictionary } = useLocale();

  const t = useCallback(
    (key: DotPaths<Dictionary[NS]>, params?: Record<string, string | number>): string => {
      const path = key as string;
      const value = resolvePath(dictionary[ns], path) ?? resolvePath(fallbackDictionary[ns], path);

      if (value === undefined) {
        if (import.meta.env.DEV) {
          console.warn(`[i18n] missing key "${String(ns)}.${path}" for locale "${locale}"`);
        }
        return path;
      }

      return interpolate(value, params);
    },
    [dictionary, fallbackDictionary, ns, locale]
  );

  return { t, locale };
}
