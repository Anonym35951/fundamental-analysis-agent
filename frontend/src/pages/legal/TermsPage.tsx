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

function TermsPage() {
  return (
    <div style={pageWrapper}>
      <h1 style={heading}>Nutzungsbedingungen</h1>
      <p style={{ color: theme.colors.textSecondary }}>Stand: Juli 2026</p>

      <h2 style={sectionHeading}>1. Leistungsbeschreibung</h2>
      <p>
        Diese Plattform stellt Aktien- und Fundamentaldaten zu von dir
        ausgewählten Unternehmen bereit, die aus öffentlich zugänglichen
        Quellen (u. a. SEC EDGAR, Yahoo Finance) aufbereitet werden. Wir
        liefern ausschließlich Daten und berechnete Kennzahlen –{" "}
        <strong>
          keine Anlageberatung, keine Kauf- oder Verkaufsempfehlung und keine
          individuelle Finanzberatung
        </strong>
        . Die Interpretation der dargestellten Daten und jede daraus
        abgeleitete Entscheidung liegt vollständig in deiner eigenen
        Verantwortung.
      </p>

      <h2 style={sectionHeading}>2. Konto &amp; Nutzung</h2>
      <p>
        Für die Nutzung der Plattform ist eine Registrierung mit gültiger
        E-Mail-Adresse erforderlich. Du bist verpflichtet, deine Zugangsdaten
        vertraulich zu behandeln und uns über eine missbräuchliche Nutzung
        deines Kontos unverzüglich zu informieren.
      </p>

      <h2 style={sectionHeading}>3. Tarife &amp; Abrechnung</h2>
      <ul style={list}>
        <li>
          Der <strong>kostenlose Plan</strong> ist auf eine begrenzte Anzahl
          Analysen pro Monat beschränkt.
        </li>
        <li>
          Der <strong>Pro-Plan</strong> wird monatlich oder jährlich über
          unseren Zahlungsdienstleister Stripe abgerechnet und verlängert
          sich automatisch, bis du kündigst.
        </li>
        <li>
          Du kannst dein Abonnement jederzeit über die Konto-Seite kündigen;
          die Kündigung wird zum Ende des laufenden Abrechnungszeitraums
          wirksam.
        </li>
      </ul>

      <h2 style={sectionHeading}>4. Widerrufsrecht für Verbraucher</h2>
      <p>
        Als Verbraucher hast du das Recht, binnen vierzehn Tagen ohne Angabe
        von Gründen den Vertrag über ein kostenpflichtiges Abonnement zu
        widerrufen. Die Widerrufsfrist beträgt vierzehn Tage ab dem Tag des
        Vertragsabschlusses. Um dein Widerrufsrecht auszuüben, musst du uns
        (Kontakt: gecenanalysis@gmail.com) mittels einer eindeutigen
        Erklärung (z.&nbsp;B. per E-Mail) über deinen Entschluss, den Vertrag
        zu widerrufen, informieren. Zur Wahrung der Widerrufsfrist reicht es
        aus, dass du die Mitteilung über die Ausübung des Widerrufsrechts vor
        Ablauf der Widerrufsfrist absendest.
      </p>
      <p>
        <strong>Folgen des Widerrufs:</strong> Wenn du den Vertrag
        widerrufst, erstatten wir dir alle Zahlungen, die wir von dir
        erhalten haben, unverzüglich und spätestens binnen vierzehn Tagen ab
        dem Tag, an dem die Mitteilung über deinen Widerruf bei uns
        eingegangen ist. Für die Rückzahlung verwenden wir dasselbe
        Zahlungsmittel, das du bei der ursprünglichen Transaktion eingesetzt
        hast.
      </p>
      <p>
        <strong>Vorzeitiges Erlöschen des Widerrufsrechts:</strong> Das
        Widerrufsrecht erlischt bei einem Vertrag über die Bereitstellung
        digitaler Dienstleistungen vorzeitig, wenn wir mit der Ausführung des
        Vertrags begonnen haben, nachdem du ausdrücklich zugestimmt hast,
        dass wir vor Ablauf der Widerrufsfrist mit der Ausführung beginnen,
        und du deine Kenntnis davon bestätigt hast, dass dein Widerrufsrecht
        mit Beginn der Vertragsausführung erlischt. Diese Zustimmung holen
        wir beim Abschluss des Abonnements ein.
      </p>

      <h2 style={sectionHeading}>5. Verfügbarkeit &amp; Datenqualität</h2>
      <p>
        Wir bemühen uns um eine hohe Verfügbarkeit und Aktualität der
        dargestellten Daten, können jedoch keine Gewähr für Richtigkeit,
        Vollständigkeit oder ständige Verfügbarkeit der Plattform oder der
        zugrunde liegenden Drittanbieter-Datenquellen übernehmen.
      </p>

      <h2 style={sectionHeading}>6. Haftung</h2>
      <p>
        Wir haften nicht für Schäden, die aus Entscheidungen entstehen, die
        auf Basis der auf dieser Plattform bereitgestellten Daten getroffen
        werden. Eine Haftung für Vorsatz und grobe Fahrlässigkeit bleibt
        hiervon unberührt.
      </p>

      <h2 style={sectionHeading}>7. Kündigung &amp; Sperrung</h2>
      <p>
        Wir behalten uns vor, Konten bei Verstoß gegen diese
        Nutzungsbedingungen oder bei missbräuchlicher Nutzung der Plattform
        zu sperren oder zu löschen.
      </p>

      <h2 style={sectionHeading}>8. Änderungen</h2>
      <p>
        Wir können diese Nutzungsbedingungen mit Wirkung für die Zukunft
        anpassen. Über wesentliche Änderungen informieren wir dich per
        E-Mail oder innerhalb der Plattform.
      </p>

      <h2 style={sectionHeading}>9. Kontakt</h2>
      <p>Fragen zu diesen Bedingungen: gecenanalysis@gmail.com</p>
    </div>
  );
}

export default TermsPage;
