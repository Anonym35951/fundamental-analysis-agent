import { Link } from "react-router-dom";
import { ArrowRight } from "lucide-react";
import { theme } from "../../components/ui/theme";

function BillingSuccessPage() {
  return (
    <div
      style={{
        minHeight: "100dvh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "20px",
        background: `linear-gradient(135deg, ${theme.colors.bgDeep}, ${theme.colors.bgDeepAlt})`,
      }}
    >
      <div
        style={{
          maxWidth: "1100px",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          gap: "22px",
        }}
      >
        <div style={heroCard}>
          <div style={successBadge}>Zahlung erfolgreich</div>

          <h1 style={heroTitle}>Willkommen im Pro-Plan 🎉</h1>

          <p style={heroText}>
            Stark, dass du den nächsten Schritt gegangen bist. Dein Upgrade war
            erfolgreich und dein Pro-Zugang ist jetzt aktiv. Ab sofort kannst du
            ComAnalysis mit mehr Tiefe, mehr Freiheit und einem deutlich
            stärkeren Analyse-Workflow nutzen.
          </p>

          <div style={highlightBox}>
            <div style={highlightTitle}>Dein Upgrade ist abgeschlossen</div>
            <div style={highlightText}>
              Wir freuen uns, dich im Pro-Plan begrüßen zu dürfen. Du hast jetzt
              Zugriff auf die erweiterte Version der Plattform und kannst direkt
              mit deinen Analysen loslegen.
            </div>
          </div>

          <div style={buttonRow}>
            <Link to="/app/dashboard" style={primaryButton}>
              Zum Dashboard
            </Link>

            <Link to="/app/account" style={secondaryButton}>
              Konto ansehen
            </Link>
          </div>
        </div>

        <div style={contentGrid}>
          <Link to="/app/analyze" style={infoCardLink}>
            <div style={sectionEyebrow}>Schritt 1</div>
            <div style={cardTitleRow}>
              <div style={cardTitle}>Starte deine erste unbegrenzte Analyse</div>
              <ArrowRight size={20} color={theme.colors.chrome} style={{ flexShrink: 0 }} />
            </div>
            <p style={cardText}>
              Ab sofort ohne Monatslimit: Wähle ein Unternehmen und lass eine
              Vollanalyse oder deine eigene Kennzahlen-Kombination laufen.
            </p>
          </Link>

          <Link to="/app/analyze" style={infoCardLink}>
            <div style={sectionEyebrow}>Schritt 2</div>
            <div style={cardTitleRow}>
              <div style={cardTitle}>Favoriten anlegen</div>
              <ArrowRight size={20} color={theme.colors.chrome} style={{ flexShrink: 0 }} />
            </div>
            <p style={cardText}>
              Markiere Unternehmen als Favorit — wir benachrichtigen dich
              automatisch per E-Mail, sobald ein neuer 10-K/10-Q-Bericht bei
              der SEC eingereicht wird.
            </p>
          </Link>
        </div>

        <div style={footerCard}>
          <div style={sectionEyebrow}>Danke für dein Upgrade</div>
          <h2 style={footerTitle}>Schön, dass du dich für Pro entschieden hast</h2>
          <p style={footerText}>
            Wir freuen uns sehr, dich als Pro-Nutzer dabei zu haben. Viel Erfolg
            bei deinen nächsten Analysen und viel Freude mit der erweiterten
            Version von ComAnalysis.
          </p>
        </div>
      </div>
    </div>
  );
}

/* styles */

const heroCard = {
  background: theme.glass.elevated.background,
  borderRadius: "28px",
  padding: "40px 32px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
  textAlign: "center" as const,
};

const successBadge = {
  display: "inline-block",
  marginBottom: "18px",
  padding: "8px 14px",
  borderRadius: "999px",
  background: theme.colors.successSoft,
  color: theme.colors.successText,
  fontWeight: 700,
  fontSize: "0.85rem",
  border: `1px solid ${theme.colors.successBorder}`,
  letterSpacing: "0.03em",
};

const heroTitle = {
  margin: "0 0 16px 0",
  fontSize: "2.7rem",
  lineHeight: 1.08,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.04em",
};

const heroText = {
  margin: "0 auto",
  maxWidth: "760px",
  color: theme.colors.textSecondary,
  fontSize: "1.08rem",
  lineHeight: 1.85,
};

const highlightBox = {
  margin: "26px auto 0 auto",
  maxWidth: "820px",
  padding: "18px 20px",
  borderRadius: "18px",
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
};

const highlightTitle = {
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  fontWeight: 800,
  marginBottom: "8px",
};

const highlightText = {
  color: theme.colors.textSecondary,
  fontSize: "0.98rem",
  lineHeight: 1.75,
};

const buttonRow = {
  display: "flex",
  justifyContent: "center",
  gap: "14px",
  flexWrap: "wrap" as const,
  marginTop: "28px",
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
  // min(420px, 100%) statt fix 420px: eine einzelne Spalte darf nicht
  // breiter als der verfügbare Platz werden, sonst überläuft die ganze
  // Seite auf jedem Handy (RESPONSIVE.md R-P0-7).
  gridTemplateColumns: "repeat(auto-fit, minmax(min(420px, 100%), 1fr))",
  gap: "22px",
};

const infoCard = {
  background: theme.colors.panel,
  borderRadius: "24px",
  padding: "28px",
  border: `1px solid ${theme.colors.borderSubtle}`,
  boxShadow: theme.glass.elevated.shadow,
};

const infoCardLink: React.CSSProperties = {
  ...infoCard,
  display: "block",
  textDecoration: "none",
  transition: `border-color ${theme.motion.fast} ${theme.motion.easing}`,
};

const sectionEyebrow = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const cardTitleRow: React.CSSProperties = {
  marginTop: "12px",
  marginBottom: "10px",
  display: "flex",
  alignItems: "flex-start",
  justifyContent: "space-between",
  gap: "12px",
};

const cardTitle = {
  fontSize: "1.4rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
  lineHeight: 1.35,
};

const cardText = {
  margin: 0,
  color: theme.colors.textSecondary,
  fontSize: "1rem",
  lineHeight: 1.8,
};

const footerCard = {
  background: theme.colors.panelAlt,
  borderRadius: "28px",
  padding: "34px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
  textAlign: "center" as const,
};

const footerTitle = {
  margin: "10px 0 14px 0",
  fontSize: "2rem",
  lineHeight: 1.15,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.03em",
};

const footerText = {
  margin: "0 auto",
  maxWidth: "760px",
  color: theme.colors.textSecondary,
  fontSize: "1.04rem",
  lineHeight: 1.85,
};

export default BillingSuccessPage;