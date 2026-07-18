import { DEFAULT_LOCALE, LOCALE_STORAGE_KEY, isSupportedLocale, type Locale } from "./config";

/** EVOLVING.md § 10: Prioritätslogik für ausgeloggte/noch nicht geladene
 * Nutzer. user.locale (eingeloggt) wird NICHT hier behandelt — das
 * übernimmt LocaleProvider nach getCurrentUser(), da detect() synchron
 * beim ersten Render laufen muss (kein Flackern) und keinen Netzwerk-
 * Zugriff hat. */

function readStoredLocale(): Locale | null {
  const stored = localStorage.getItem(LOCALE_STORAGE_KEY);
  return isSupportedLocale(stored) ? stored : null;
}

function readBrowserLocale(): Locale | null {
  const candidates = navigator.languages?.length ? navigator.languages : [navigator.language];
  for (const candidate of candidates) {
    if (!candidate) continue;
    const prefix = candidate.split("-")[0].toLowerCase();
    if (isSupportedLocale(prefix)) return prefix;
  }
  return null;
}

/** localStorage → navigator.language(s) → Default. Reine Geräte-/Browser-
 * Ermittlung, läuft synchron und ohne Netzwerk. */
export function detectInitialLocale(): Locale {
  return readStoredLocale() ?? readBrowserLocale() ?? DEFAULT_LOCALE;
}

export function persistLocale(locale: Locale): void {
  localStorage.setItem(LOCALE_STORAGE_KEY, locale);
}
