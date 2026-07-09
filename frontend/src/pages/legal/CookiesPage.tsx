import type { CSSProperties } from "react";
import { theme } from "../../components/ui/theme";

const pageWrapper: CSSProperties = {
  maxWidth: "760px",
  margin: "0 auto",
  padding: "64px 24px 96px",
  color: theme.colors.textPrimary,
  lineHeight: 1.7,
};

const heading: CSSProperties = {
  fontSize: "2rem",
  marginBottom: "8px",
};

const sectionHeading: CSSProperties = {
  fontSize: "1.2rem",
  marginTop: "32px",
  marginBottom: "8px",
};

const list: CSSProperties = {
  paddingLeft: "20px",
  margin: "8px 0",
};

const resetButton: CSSProperties = {
  marginTop: "8px",
  padding: "11px 22px",
  borderRadius: theme.radius.pill,
  border: `1px solid ${theme.colors.chromeBorder}`,
  background: theme.colors.panelAlt,
  color: theme.colors.textPrimary,
  fontWeight: 600,
  cursor: "pointer",
};

function handleResetConsent() {
  localStorage.removeItem("analytics_consent");
  window.location.reload();
}

function CookiesPage() {
  return (
    <div style={pageWrapper}>
      <h1 style={heading}>Cookies &amp; Analyse</h1>
      <p style={{ color: theme.colors.textSecondary }}>Stand: Juli 2026</p>

      <h2 style={sectionHeading}>1. Übersicht</h2>
      <p>
        Diese Seite erklärt, welche Analyse-Tools ComAnalysis einsetzt und
        welche Daten dabei anfallen. Aktuell setzen wir ausschließlich
        Plausible Analytics ein.
      </p>

      <h2 style={sectionHeading}>2. Plausible Analytics</h2>
      <p>
        Plausible ist ein datenschutzfreundlicher Analyse-Dienst. Im
        Unterschied zu klassischen Tracking-Tools setzt Plausible{" "}
        <strong>keine Cookies</strong> und verarbeitet{" "}
        <strong>keine personenbezogenen Daten</strong>. Es werden lediglich
        anonymisierte, aggregierte Kennzahlen erfasst:
      </p>
      <ul style={list}>
        <li>Aufgerufene Seite und Referrer (woher der Besuch kam)</li>
        <li>Grober Gerätetyp/Browser, abgeleitet aus dem User-Agent</li>
        <li>
          Land, abgeleitet aus der IP-Adresse — die IP-Adresse selbst wird
          nicht gespeichert
        </li>
      </ul>
      <p>
        Es findet kein seitenübergreifendes Tracking statt und einzelne
        Besucher werden nicht wiedererkannt. Das Script wird erst geladen,
        nachdem du im Cookie-Banner zugestimmt hast.
      </p>

      <h2 style={sectionHeading}>3. Zukünftige Cookies</h2>
      <p>
        Sollten wir künftig echte Cookies einsetzen (z. B. für eine sicherere
        Anmeldeverwaltung), aktualisieren wir diese Seite und fragen deine
        Zustimmung erneut über das Banner ab.
      </p>

      <h2 style={sectionHeading}>4. Widerruf</h2>
      <p>
        Du kannst deine Zustimmung jederzeit widerrufen. Da Plausible keine
        Cookies setzt, gibt es dabei nichts im klassischen Sinn zu löschen —
        ein Widerruf sorgt lediglich dafür, dass das Script nicht mehr
        nachgeladen wird.
      </p>
      <button type="button" style={resetButton} onClick={handleResetConsent}>
        Einstellungen ändern
      </button>
    </div>
  );
}

export default CookiesPage;
