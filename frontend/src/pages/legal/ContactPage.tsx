import type { CSSProperties } from "react";
import { theme } from "../../components/ui/theme";
import SupportForm from "../../components/support/SupportForm";

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

const card: CSSProperties = {
  marginTop: "24px",
  padding: "24px",
  borderRadius: theme.radius.lg,
  border: `1px solid ${theme.glass.subtle.border}`,
  background: theme.glass.subtle.background,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
};

function ContactPage() {
  return (
    <div style={pageWrapper}>
      <h1 style={heading}>Kontakt</h1>
      <p style={{ color: theme.colors.textSecondary }}>
        Du hast Fragen zu deinem Konto, einer Analyse oder deinem
        Abonnement? Wir helfen dir gerne weiter.
      </p>

      <div style={card}>
        <SupportForm />
      </div>

      <div style={{ ...card, marginTop: "16px" }}>
        <p style={{ margin: 0 }}>
          <strong>E-Mail:</strong>{" "}
          <a href="mailto:gecenanalysis@gmail.com" style={{ color: theme.colors.chrome }}>
            gecenanalysis@gmail.com
          </a>
        </p>
        <p style={{ margin: "8px 0 0", color: theme.colors.textSecondary }}>
          Wir antworten in der Regel innerhalb von 1–2 Werktagen.
        </p>
      </div>

      <p style={{ marginTop: "32px", color: theme.colors.textSecondary }}>
        Für rechtliche Angaben siehe das{" "}
        <a href="/legal/imprint" style={{ color: theme.colors.chrome }}>Impressum</a>, für Informationen zur
        Datenverarbeitung siehe die{" "}
        <a href="/legal/privacy" style={{ color: theme.colors.chrome }}>Datenschutzerklärung</a>.
      </p>
    </div>
  );
}

export default ContactPage;
