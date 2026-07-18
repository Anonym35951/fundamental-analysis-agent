/** EVOLVING.md § Internationalisierung, § 13: zentrale i18n-Konfiguration.
 * Neue Sprache hinzufügen? Siehe § 30 "Adding a New Language Checklist" —
 * hier nur SUPPORTED_LOCALES/INTL_LOCALES ergänzen, kein Architekturumbau. */

export const SUPPORTED_LOCALES = ["de", "en"] as const;
export type Locale = (typeof SUPPORTED_LOCALES)[number];

export const DEFAULT_LOCALE: Locale = "de";

/** Persistenz-Key für die Geräte-/Browser-Präferenz — analog "theme-mode"
 * (components/ui/ThemeModeContext.tsx). */
export const LOCALE_STORAGE_KEY = "app-locale";

/** BCP-47-Tag je Locale für Intl.* (Zahlen/Datum/Prozent) — strikt getrennt
 * von der Anzeigewährung, die immer vom Backend als ISO-Code kommt. */
export const INTL_LOCALES: Record<Locale, string> = {
  de: "de-DE",
  en: "en-US",
};

export function isSupportedLocale(value: unknown): value is Locale {
  return typeof value === "string" && (SUPPORTED_LOCALES as readonly string[]).includes(value);
}
