import type { Step } from "react-joyride";

/** Eine Tour-Station: normale react-joyride-Step-Felder plus eine eigene
 * `route`, auf die vor dem Anzeigen dieses Schritts navigiert werden muss
 * (react-joyride selbst kennt keine Routen, nur DOM-Ziele), sowie ein
 * optionales `data`-Feld für Koordination mit Seiten-lokalem State (siehe
 * `requiredTab` beim Schritt "Eigene Analyse erstellen"). */
export type TourStep = Step & { route: string; data?: Record<string, unknown> };

/** Reine Navigations-Hinweise, bewusst neutral formuliert (kein Kauf-/
 * Rendite-Bezug) - siehe Positionierungs-Leitplanke des Produkts. */
export const tourSteps: TourStep[] = [
  {
    route: "/app/dashboard",
    target: '[data-tour="dashboard-cta"]',
    title: "Starte deine erste Analyse",
    content:
      "Von hier aus gelangst du direkt zur Analyse-Seite. Wähle ein Unternehmen und starte eine strukturierte Fundamentalanalyse.",
    placement: "bottom",
  },
  {
    route: "/app/dashboard",
    target: '[data-tour="dashboard-analysis-history"]',
    title: "Analyse-Historie",
    content:
      "Deine bisherigen Analysen werden hier gespeichert. Über das Verlauf-Symbol kannst du eine frühere Analyse mit denselben Einstellungen erneut ausführen.",
    placement: "bottom",
  },
  {
    route: "/app/dashboard",
    target: '[data-tour="sidebar-favorites"]',
    title: "Favoriten",
    content:
      "Markiere ein Unternehmen als Favorit, um es hier in der Seitenleiste für den schnellen Zugriff zu speichern.",
    placement: "right",
    // Auf Mobile ist die Sidebar per Default ein geschlossener Off-Canvas-
    // Drawer — AppLayout.tsx liest dieses Flag und öffnet den Drawer
    // automatisch, solange dieser Schritt aktiv ist (und schließt ihn
    // danach wieder), statt den Schritt zu überspringen.
    data: { requireSidebarOpen: true },
  },
  {
    route: "/app/analyze",
    target: '[data-tour="analyze-mode-tabs"]',
    title: "Analyse-Modus wählen",
    content:
      "Nutze einen vordefinierten Analyse-Modus oder wechsle zu \"Individuell\", um deine eigenen Kennzahlen frei zu kombinieren.",
    placement: "bottom",
  },
  {
    route: "/app/analyze",
    target: '[data-tour="analyze-custom-builder"]',
    title: "Eigene Analyse erstellen",
    content:
      "Im Modus \"Individuell\" kombinierst du beliebige Kennzahlen zu deiner eigenen wiederverwendbaren Analyse-Vorlage.",
    placement: "bottom",
    data: { requiredTab: "individuell" },
  },
  {
    route: "/app/compare",
    target: '[data-tour="compare-metrics-picker"]',
    title: "Kennzahlen auswählen",
    content:
      "Jede ausgewählte Kennzahl pro Unternehmen zählt im Free-Plan gegen dein monatliches Kontingent. Du kannst dabei so viele Kennzahlen frei kombinieren, wie du möchtest — es gibt keine Obergrenze.",
    placement: "bottom",
  },
  {
    route: "/app/compare",
    target: '[data-tour="compare-symbol-input"]',
    title: "Unternehmen vergleichen",
    content:
      "Hier fügst du mehrere Unternehmen hinzu, um ihre Kennzahlen direkt nebeneinander zu vergleichen.",
    placement: "bottom",
  },
  {
    route: "/app/account",
    target: '[data-tour="account-profile-section"]',
    title: "Dein Profil",
    content:
      "Hier kannst du dein Profil vervollständigen (Benutzername, Name, Alter) und dein Konto verwalten.",
    placement: "top",
  },
  {
    route: "/app/billing",
    target: '[data-tour="billing-plans"]',
    title: "Dein Plan",
    content:
      "Hier siehst du die Details zu deinem aktuellen Plan und die Upgrade-Optionen im Überblick.",
    placement: "top",
  },
];
