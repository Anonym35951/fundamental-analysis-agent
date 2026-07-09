let isLoaded = false;

/** Laedt das Plausible-Analytics-Script - nur aufrufen, nachdem der Nutzer
 * im Cookie-Consent-Banner zugestimmt hat (siehe CookieConsentBanner.tsx).
 * No-op wenn nicht konfiguriert (z.B. lokale Entwicklung) oder bereits
 * geladen. Plausible selbst setzt keine Cookies und speichert keine
 * personenbezogenen Daten - trotzdem hinter Consent gated, fuer
 * Transparenz und als Grundlage fuer kuenftige First-Party-Cookies. */
export function loadPlausibleScript() {
  if (isLoaded) return;

  const domain = import.meta.env.VITE_PLAUSIBLE_DOMAIN;
  const apiHost = import.meta.env.VITE_PLAUSIBLE_API_HOST;
  if (!domain || !apiHost) return;

  const script = document.createElement("script");
  script.defer = true;
  script.setAttribute("data-domain", domain);
  script.src = `${apiHost}/js/script.js`;
  document.head.appendChild(script);
  isLoaded = true;
}
