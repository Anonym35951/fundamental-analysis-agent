import { Link } from "react-router-dom";
import { theme } from "../ui/theme";
import { useTranslation } from "../../i18n/useTranslation";

type FooterProps = {
  variant?: "public" | "auth" | "app";
};

function Footer({ variant = "public" }: FooterProps) {
  const { t } = useTranslation("nav");

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
            {t("footer.copyright", { year: new Date().getFullYear() })}
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
              {t("footer.glossaryLink")}
            </a>

            <Link to="/pricing" style={linkStyle}>
              {t("footer.pricingLink")}
            </Link>

            <Link to="/legal/privacy" style={linkStyle}>
              {t("footer.privacyLink")}
            </Link>

            <Link to="/legal/cookies" style={linkStyle}>
              {t("footer.cookiesLink")}
            </Link>

            <Link to="/legal/terms" style={linkStyle}>
              {t("footer.termsLink")}
            </Link>

            <Link to="/legal/imprint" style={linkStyle}>
              {t("footer.imprintLink")}
            </Link>

            <Link to="/legal/contact" style={linkStyle}>
              {t("footer.contactLink")}
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