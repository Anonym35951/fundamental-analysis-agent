import { useCallback, useEffect, useMemo, useState, type ReactNode } from "react";
import { LocaleContext, type LocaleContextValue } from "./localeContextValue";
import { isSupportedLocale, type Locale } from "./config";
import { detectInitialLocale, persistLocale } from "./detect";
import { de, type Dictionary } from "./locales/de";
import { getCurrentUser } from "../api/auth";

/** EVOLVING.md § 13: Sprachwechsel ist reiner React-State → Re-Render ohne
 * Reload. Sämtlicher App-State (Compare-Auswahl, laufende Analysis-Jobs,
 * Workspace, Favoriten, Router-Position, Formulareingaben) bleibt per
 * Konstruktion erhalten, weil die Locale nirgends in Request-Bau oder
 * State-Keys einfließt (siehe Paritätstests § 16/28). */
export function LocaleProvider({ children }: { children: ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(detectInitialLocale);
  const [enDictionary, setEnDictionary] = useState<Dictionary | null>(null);

  // DE ist eager im Bundle (heutiger Zustand ohnehin), EN wird erst bei
  // tatsächlicher Aktivierung nachgeladen (§ 14 Lazy-Loading-Entscheidung).
  useEffect(() => {
    if (locale !== "en" || enDictionary) return;
    let cancelled = false;
    import("./locales/en").then((mod) => {
      if (!cancelled) setEnDictionary(mod.en);
    });
    return () => {
      cancelled = true;
    };
  }, [locale, enDictionary]);

  useEffect(() => {
    document.documentElement.lang = locale;
  }, [locale]);

  const setLocale = useCallback((next: Locale) => {
    setLocaleState(next);
    persistLocale(next);
  }, []);

  // EVOLVING.md § 10: eine gesetzte Konto-Praeferenz gewinnt ueber die
  // Geraete-/Browser-Wahl. Laeuft sowohl beim ersten Laden mit bereits
  // gueltigem Token (Reload/neuer Tab) als auch nach einem frischen Login
  // (app:login-Event, siehe LoginPage.tsx). NULL-Praeferenz => Geraetewahl
  // bleibt unangetastet, kein Schreibzugriff hier (nur Lesen).
  useEffect(() => {
    function syncFromAccount() {
      if (!localStorage.getItem("access_token")) return;
      getCurrentUser()
        .then((user) => {
          if (isSupportedLocale(user.locale)) {
            setLocale(user.locale);
          }
        })
        .catch(() => {
          // Nicht eingeloggt / Netzwerkfehler - Geraetewahl bleibt bestehen.
        });
    }

    syncFromAccount();
    window.addEventListener("app:login", syncFromAccount);
    return () => window.removeEventListener("app:login", syncFromAccount);
  }, [setLocale]);

  // Bis das EN-Bundle geladen ist bleibt DE sichtbar — nie rohe Keys oder
  // ein leerer Zwischenzustand (§ 13).
  const ready = locale === "de" || enDictionary !== null;
  const activeDictionary: Dictionary = locale === "en" && enDictionary ? enDictionary : de;

  const value = useMemo<LocaleContextValue>(
    () => ({ locale, setLocale, ready, dictionary: activeDictionary, fallbackDictionary: de }),
    [locale, setLocale, ready, activeDictionary]
  );

  return <LocaleContext.Provider value={value}>{children}</LocaleContext.Provider>;
}
