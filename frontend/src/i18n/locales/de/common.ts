/** EVOLVING.md § Internationalisierung, I18N-007 (Phase 3): generisches
 * UI-Chrome, das app-weit wiederverwendet wird (Toast, Modal,
 * ErrorBoundary, Cookie-Consent-Banner). Strings hier 1:1 aus dem
 * bisherigen Code übernommen (Paritätsgebot § 2). */
export const common = {
  toast: {
    dismissAriaLabel: "Benachrichtigung schließen",
  },
  modal: {
    closeAriaLabel: "Schließen",
  },
  errorBoundary: {
    title: "Etwas ist schiefgelaufen.",
    description:
      "Diese Ansicht konnte nicht geladen werden. Bitte lade die Seite neu. Falls das Problem weiterhin besteht, kontaktiere den Support.",
    reloadButton: "Seite neu laden",
  },
  cookieConsent: {
    brandStrong: "Cloudflare Web Analytics",
    mobileSuffix: "— cookielos, keine PII.",
    mobileLinkLabel: "Details",
    desktopPrefix: "Wir nutzen",
    desktopMiddle:
      "— datenschutzfreundlich, ohne Cookies und ohne personenbezogene Daten. Details in unseren",
    desktopLinkLabel: "Cookie-Hinweisen",
    rejectButton: "Ablehnen",
    acceptButton: "Zustimmen",
  },
} as const;
