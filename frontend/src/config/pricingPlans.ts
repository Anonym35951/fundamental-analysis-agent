/** Einzige Quelle für die Free/Pro-Plandaten (Preise, Feature-Liste,
 * Rabatt-Texte) — von BillingPage.tsx (authentifizierter Checkout) UND
 * PricingPage.tsx (öffentliche Preis-Seite) importiert, damit beide niemals
 * auseinanderlaufen können (siehe LAUNCH.md P1-4: vorher gab es gar keine
 * öffentliche Preis-Information, jetzt darf es keine zweite, abweichende
 * Kopie geben). Ändere Preise/Features nur hier. */

export const FREE_PLAN = {
  name: "Free",
  tag: "Einstieg",
  priceLabel: "0 €",
  description:
    "Zum Einstieg: Unternehmen selbst analysieren und den strukturierten Prozess kennenlernen.",
  features: [
    "50 Analyse-Einheiten pro Monat",
    "Alle Kennzahlen und Analysemodi verfügbar",
    "1 gespeicherte eigene Analyse-Definition",
    "Analyse-Historie und Favoriten",
  ],
} as const;

export const PRO_PLAN = {
  name: "Pro",
  tag: "Empfohlen",
  monthlyPriceLabel: "50 € / Monat",
  yearlyPriceLabel: "500 € / Jahr",
  yearlyOldPriceLabel: "600 €",
  savingsBadge: "2 Monate gratis",
  savingsText: "100 € günstiger pro Jahr",
  yearlySelectionNote: "Jährlich · Spare 16,7 %",
  monthlyHint: "Monatlich flexibel kündbar",
  description:
    "Für alle, die regelmäßig selbst analysieren und sich eine unbegrenzte Bibliothek eigener Analyse-Logiken aufbauen wollen.",
  features: [
    "Unbegrenzte Analysen — kein Monatslimit",
    "Alle Kennzahlen und Analysemodi verfügbar",
    "Beliebig viele gespeicherte eigene Analyse-Definitionen",
    "Analyse-Historie und Favoriten",
  ],
} as const;
