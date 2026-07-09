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

function PrivacyPage() {
  return (
    <div style={pageWrapper}>
      <h1 style={heading}>Datenschutzerklärung</h1>
      <p style={{ color: theme.colors.textSecondary }}>Stand: Juli 2026</p>

      <h2 style={sectionHeading}>1. Verantwortlicher</h2>
      <p>
        Efe Gecen
        <br />
        [Bitte vollständige Postanschrift ergänzen]
        <br />
        E-Mail: gecenanalysis@gmail.com
      </p>

      <h2 style={sectionHeading}>2. Welche Daten wir verarbeiten</h2>
      <ul style={list}>
        <li>
          <strong>Account-Daten:</strong> E-Mail-Adresse und Passwort (als
          sicherer Hash gespeichert, niemals im Klartext) bei Registrierung
          und Login.
        </li>
        <li>
          <strong>Nutzungsdaten:</strong> angeforderte Aktienanalysen
          (Symbol, Analysemodus, Zeitstempel) und die Anzahl monatlich
          genutzter Analysen, soweit für die Plan-Limitierung erforderlich.
        </li>
        <li>
          <strong>Zahlungsdaten:</strong> Abonnement- und Zahlungsdaten werden
          ausschließlich von unserem Zahlungsdienstleister Stripe verarbeitet.
          Wir selbst speichern keine vollständigen Kreditkartendaten, sondern
          lediglich die von Stripe übermittelte Kunden- und Abonnement-ID.
        </li>
        <li>
          <strong>Technische Daten:</strong> Zugriffstoken (JWT) wird nach dem
          Login lokal im Browser (localStorage) gespeichert, um dich
          eingeloggt zu halten.
        </li>
      </ul>

      <h2 style={sectionHeading}>3. Zweck der Verarbeitung</h2>
      <p>
        Wir verarbeiten die genannten Daten, um dir die Nutzung der Plattform
        (Registrierung, Login, Durchführung von Aktienanalysen, Verwaltung
        deines Abonnements) zu ermöglichen, dein Nutzungslimit korrekt
        anzuwenden und dich per E-Mail über sicherheits- und
        abrechnungsrelevante Ereignisse (z. B. fehlgeschlagene Zahlung,
        Plan-Änderung) zu informieren.
      </p>

      <h2 style={sectionHeading}>4. Eingesetzte Drittanbieter</h2>
      <ul style={list}>
        <li>
          <strong>Stripe</strong> – Zahlungsabwicklung und
          Abonnementverwaltung.
        </li>
        <li>
          <strong>SMTP-E-Mail-Versand</strong> – Versand von
          Transaktions-E-Mails (z. B. Zahlungsbenachrichtigungen).
        </li>
        <li>
          <strong>Externe Finanzdatenquellen</strong> (SEC EDGAR, Yahoo
          Finance, Alpha Vantage, FRED, SimFin) – werden ausschließlich zum
          Abruf öffentlicher Unternehmens- und Marktdaten zu von dir
          ausgewählten Aktien-Symbolen genutzt. Dabei werden keine
          personenbezogenen Daten an diese Anbieter übermittelt.
        </li>
        <li>
          <strong>Plausible Analytics</strong> – datenschutzfreundlicher,
          cookieloser Analyse-Dienst zur Auswertung der Website-Nutzung.
          Verarbeitet keine personenbezogenen Daten und erkennt einzelne
          Besucher nicht wieder; wird erst nach deiner Zustimmung im
          Cookie-Banner geladen. Details siehe{" "}
          <a href="/legal/cookies" style={{ color: "inherit" }}>
            Cookies &amp; Analyse
          </a>
          . Rechtsgrundlage: Einwilligung (Art. 6 Abs. 1 lit. a DSGVO).
        </li>
      </ul>

      <h2 style={sectionHeading}>5. Speicherdauer</h2>
      <p>
        Account-Daten werden gespeichert, solange dein Konto aktiv ist. Nach
        Löschung deines Kontos werden personenbezogene Daten innerhalb
        angemessener Frist gelöscht, soweit keine gesetzlichen
        Aufbewahrungspflichten (z. B. handels- oder steuerrechtlich für
        Zahlungsbelege) entgegenstehen.
      </p>

      <h2 style={sectionHeading}>6. Deine Rechte</h2>
      <p>
        Du hast jederzeit das Recht auf Auskunft, Berichtigung, Löschung und
        Einschränkung der Verarbeitung deiner personenbezogenen Daten sowie
        das Recht auf Datenübertragbarkeit und Widerspruch. Wende dich dazu
        an: gecenanalysis@gmail.com.
      </p>

      <h2 style={sectionHeading}>7. Kontakt</h2>
      <p>
        Bei Fragen zum Datenschutz erreichst du uns unter
        gecenanalysis@gmail.com.
      </p>
    </div>
  );
}

export default PrivacyPage;
