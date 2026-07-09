import { Link } from "react-router-dom";
import { theme } from "../ui/theme";

type FooterProps = {
  variant?: "public" | "auth" | "app";
};

function Footer({ variant = "public" }: FooterProps) {
  return (
    <footer
      style={{
        position: "relative",
        zIndex: 1,
        borderTop: "1px solid rgba(148, 163, 184, 0.1)",
        background: variant === "app" ? theme.colors.bgDeep : "transparent",
      }}
    >
      <div
        style={{
          maxWidth: "1200px",
          margin: "0 auto",
          padding: "26px 24px",
          display: "flex",
          flexDirection: "column",
          gap: "16px",
        }}
      >
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: "16px",
            flexWrap: "wrap",
          }}
        >
          <div
            style={{
              color: theme.colors.textMuted,
              fontSize: "0.95rem",
            }}
          >
            © {new Date().getFullYear()} ComAnalysis
          </div>

          <nav
            style={{
              display: "flex",
              gap: "18px",
              flexWrap: "wrap",
            }}
          >
            {/* Plain <a>, nicht <Link>: /glossar ist eine statisch vorgerenderte
             * Seite außerhalb des React-Router-Baums (siehe scripts/generate-glossary.ts),
             * ein Client-seitiger Router-Wechsel würde dort nur einen 404 im SPA-Kontext zeigen. */}
            <a href="/glossar" style={linkStyle}>
              Kennzahlen-Glossar
            </a>

            <Link to="/legal/privacy" style={linkStyle}>
              Datenschutz
            </Link>

            <Link to="/legal/cookies" style={linkStyle}>
              Cookies
            </Link>

            <Link to="/legal/terms" style={linkStyle}>
              AGB
            </Link>

            <Link to="/legal/imprint" style={linkStyle}>
              Impressum
            </Link>

            <Link to="/legal/contact" style={linkStyle}>
              Kontakt
            </Link>
          </nav>
        </div>
      </div>
    </footer>
  );
}

const linkStyle = {
  textDecoration: "none",
  color: theme.colors.textSecondary,
  fontSize: "0.95rem",
  fontWeight: 500,
};

export default Footer;