import { Link } from "react-router-dom";
import { theme } from "../../components/ui/theme";

function BillingCancelPage() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "28px",
        padding: "10px 6px 20px",
      }}
    >
      <section style={heroSection}>
        <div style={heroBadge}>Zahlung abgebrochen</div>

        <h1
          style={{
            margin: "0 0 14px 0",
            fontSize: "3rem",
            lineHeight: 1.05,
            letterSpacing: "-0.045em",
            color: theme.colors.textPrimary,
          }}
        >
          Dein Upgrade wurde nicht abgeschlossen
        </h1>

        <p
          style={{
            margin: 0,
            maxWidth: "820px",
            color: theme.colors.textPrimary,
            fontSize: "1.12rem",
            lineHeight: 1.9,
          }}
        >
          Schade, dass du dein Upgrade gerade nicht abgeschlossen hast. Kein
          Problem: Dein aktueller Zugang bleibt bestehen und es wurde nichts
          verändert. Wir würden uns freuen, dich jederzeit wieder beim Upgrade
          begrüßen zu dürfen, sobald du bereit bist.
        </p>

        <div style={heroMessageBox}>
          <div style={heroMessageTitle}>Du bist jederzeit willkommen</div>
          <div style={heroMessageText}>
            Wenn du dich später doch für Pro entscheidest, kannst du den
            Upgrade-Prozess in wenigen Klicks erneut starten. Wir freuen uns,
            wenn du wieder zurückkommst.
          </div>
        </div>

        <div
          style={{
            display: "flex",
            gap: "14px",
            flexWrap: "wrap",
            marginTop: "26px",
          }}
        >
          <Link to="/app/billing" style={primaryButton}>
            Zurück zur Billing-Seite
          </Link>

          <Link to="/app/dashboard" style={secondaryButton}>
            Zum Dashboard
          </Link>
        </div>
      </section>

      <section style={contentGrid}>
        <div style={infoCard}>
          <div style={sectionEyebrow}>Was bedeutet das?</div>
          <div style={cardTitle}>Dein Plan wurde nicht geändert</div>
          <p style={cardText}>
            Der Checkout wurde vor dem Abschluss beendet. Dadurch bleibt dein
            bisheriger Tarif aktiv und es wurde kein neues Upgrade
            freigeschaltet.
          </p>
        </div>

        <div style={infoCard}>
          <div style={sectionEyebrow}>Du kannst jederzeit zurückkommen</div>
          <div style={cardTitle}>Upgrade später erneut starten</div>
          <p style={cardText}>
            Wenn du Pro doch aktivieren möchtest, kannst du den Checkout einfach
            erneut über die Billing-Seite aufrufen und den gewünschten Plan noch
            einmal auswählen.
          </p>
        </div>
      </section>

      <section style={helpSection}>
        <div style={sectionEyebrow}>Nächster Schritt</div>
        <h2 style={helpTitle}>Möchtest du es noch einmal versuchen?</h2>
        <p style={helpText}>
          Wir würden uns freuen, wenn du ComAnalysis später doch mit Pro nutzen
          möchtest. Kehre einfach zur Billing-Seite zurück und starte den
          Checkout erneut. Dort kannst du zwischen monatlicher und jährlicher
          Abrechnung wählen.
        </p>

        <div
          style={{
            display: "flex",
            gap: "14px",
            flexWrap: "wrap",
            marginTop: "24px",
          }}
        >
          <Link to="/app/billing" style={helpPrimaryButton}>
            Erneut zum Upgrade
          </Link>

          <Link to="/app/account" style={helpSecondaryButton}>
            Konto verwalten
          </Link>
        </div>
      </section>
    </div>
  );
}

/* styles */

const heroSection = {
  background: theme.glass.elevated.background,
  borderRadius: "28px",
  padding: "34px 34px 36px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
};

const heroBadge = {
  display: "inline-block",
  marginBottom: "16px",
  padding: "8px 12px",
  borderRadius: "999px",
  background: theme.colors.dangerSoft,
  border: `1px solid ${theme.colors.dangerBorder}`,
  color: theme.colors.dangerText,
  fontSize: "0.86rem",
  fontWeight: 700,
  letterSpacing: "0.03em",
};

const heroMessageBox = {
  marginTop: "22px",
  padding: "18px 20px",
  borderRadius: "18px",
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  maxWidth: "820px",
};

const heroMessageTitle = {
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  fontWeight: 800,
  marginBottom: "8px",
};

const heroMessageText = {
  color: theme.colors.textPrimary,
  fontSize: "0.98rem",
  lineHeight: 1.75,
};

const primaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "14px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.onChrome,
  fontWeight: 800,
  fontSize: "1rem",
  boxShadow: theme.glass.elevated.shadow,
};

const secondaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "14px",
  background: theme.colors.panelAlt,
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "1rem",
  border: `1px solid ${theme.colors.border}`,
};

const contentGrid = {
  display: "grid",
  // auto-fit statt fix "1fr 1fr": kollabiert auf schmalen Screens von selbst
  // auf 1 Spalte statt zwei Info-Karten mit Fließtext zu quetschen
  // (RESPONSIVE.md R-P1-3).
  gridTemplateColumns: "repeat(auto-fit, minmax(min(300px, 100%), 1fr))",
  gap: "22px",
};

const infoCard = {
  background: theme.colors.panel,
  borderRadius: "24px",
  padding: "28px",
  border: `1px solid ${theme.colors.borderSubtle}`,
  boxShadow: theme.glass.elevated.shadow,
};

const sectionEyebrow = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const cardTitle = {
  marginTop: "12px",
  marginBottom: "10px",
  fontSize: "1.4rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
  lineHeight: 1.35,
};

const cardText = {
  margin: 0,
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  lineHeight: 1.8,
};

const helpSection = {
  background: theme.colors.panelAlt,
  borderRadius: "28px",
  padding: "34px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
};

const helpTitle = {
  margin: "10px 0 14px 0",
  fontSize: "2.1rem",
  lineHeight: 1.15,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.03em",
};

const helpText = {
  margin: 0,
  maxWidth: "760px",
  color: theme.colors.textPrimary,
  fontSize: "1.04rem",
  lineHeight: 1.85,
};

const helpPrimaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "15px 18px",
  borderRadius: "14px",
  background: theme.colors.chromeStrong,
  color: theme.colors.onChrome,
  fontWeight: 800,
  fontSize: "1rem",
};

const helpSecondaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "15px 18px",
  borderRadius: "14px",
  background: "transparent",
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "1rem",
  border: `1px solid ${theme.colors.border}`,
};

export default BillingCancelPage;