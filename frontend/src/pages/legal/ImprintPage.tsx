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

function ImprintPage() {
  return (
    <div style={pageWrapper}>
      <h1 style={heading}>Impressum</h1>
      <p style={{ color: theme.colors.textSecondary }}>Angaben gemäß § 5 DDG</p>

      <h2 style={sectionHeading}>Verantwortlich für den Inhalt</h2>
      <p>
        Efe Gecen
        <br />
        Handgasse 2
        <br />
        97318 Kitzingen
        <br />
        Deutschland
      </p>

      <h2 style={sectionHeading}>Kontakt</h2>
      <p>E-Mail: kontakt@comanalysis.de</p>

      <h2 style={sectionHeading}>Umsatzsteuer-ID</h2>
      <p>
        Es wird derzeit keine Umsatzsteuer-Identifikationsnummer ausgewiesen.
      </p>

      <h2 style={sectionHeading}>Haftung für Inhalte</h2>
      <p>
        Als Diensteanbieter sind wir gemäß § 7 Abs. 1 DDG für eigene Inhalte
        auf diesen Seiten nach den allgemeinen Gesetzen verantwortlich. Die
        auf dieser Plattform dargestellten Finanzkennzahlen und Analysen
        beruhen auf öffentlich zugänglichen Datenquellen (u. a. SEC EDGAR,
        Yahoo Finance) und werden ohne Gewähr für Richtigkeit, Vollständigkeit
        oder Aktualität bereitgestellt. Sie stellen keine Anlageberatung,
        keine Kauf- oder Verkaufsempfehlung und keine Finanzberatung dar. Jede
        Interpretation der dargestellten Daten und daraus abgeleitete
        Entscheidungen liegen ausschließlich in der Verantwortung der
        nutzenden Person.
      </p>

      <h2 style={sectionHeading}>Haftung für Links</h2>
      <p>
        Unser Angebot enthält gegebenenfalls Links zu externen Webseiten
        Dritter, auf deren Inhalte wir keinen Einfluss haben. Für die Inhalte
        der verlinkten Seiten ist stets der jeweilige Anbieter verantwortlich.
      </p>
    </div>
  );
}

export default ImprintPage;
