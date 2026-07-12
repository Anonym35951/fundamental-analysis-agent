let isLoaded = false;

/** Laedt das Cloudflare-Web-Analytics-Beacon-Script - nur aufrufen, nachdem
 * der Nutzer im Cookie-Consent-Banner zugestimmt hat (siehe
 * CookieConsentBanner.tsx). No-op wenn nicht konfiguriert (z.B. lokale
 * Entwicklung) oder bereits geladen. Cloudflare Web Analytics setzt keine
 * Cookies und speichert keine personenbezogenen Daten - trotzdem hinter
 * Consent gated, fuer Transparenz und als Grundlage fuer kuenftige
 * First-Party-Cookies. */
export function loadWebAnalyticsScript() {
  if (isLoaded) return;

  const token = import.meta.env.VITE_CF_BEACON_TOKEN;
  if (!token) return;

  const script = document.createElement("script");
  script.defer = true;
  script.src = "https://static.cloudflareinsights.com/beacon.min.js";
  script.setAttribute("data-cf-beacon", JSON.stringify({ token }));
  document.head.appendChild(script);
  isLoaded = true;
}
