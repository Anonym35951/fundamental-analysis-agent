import { Link } from "react-router-dom";
import { theme } from "../ui/theme";

type HeaderProps = {
  variant?: "public" | "auth";
};

function Header({ variant = "public" }: HeaderProps) {
  const hasToken = Boolean(localStorage.getItem("access_token"));

  return (
    <header
      style={{
        position: "sticky",
        top: 0,
        zIndex: 100,
        padding: "18px 24px",
      }}
    >
      <div
        style={{
          maxWidth: "1280px",
          margin: "0 auto",
          display: "flex",
          flexWrap: "wrap",
          rowGap: "10px",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 14px",
          borderRadius: theme.radius.pill,
          background: theme.glass.subtle.background,
          border: `1px solid ${theme.glass.subtle.border}`,
          backdropFilter: `blur(${theme.glass.subtle.blur})`,
          WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
        }}
      >
        <Link
          to={hasToken ? "/app/dashboard" : "/landing"}
          style={{
            textDecoration: "none",
            color: theme.colors.textPrimary,
            fontWeight: 900,
            fontSize: "1.5rem",
            letterSpacing: "-0.04em",
            padding: "0 10px",
          }}
        >
          ComAnalysis
        </Link>

        {/* Navigation */}
        <nav
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "4px",
            alignItems: "center",
            padding: "4px",
            borderRadius: theme.radius.pill,
            background: "transparent",
          }}
        >
          {/* PUBLIC (logged in) */}
          {variant === "public" && hasToken && (
            <Link to="/app/dashboard" style={pillCta}>
              Zum Dashboard
            </Link>
          )}

          {/* PUBLIC (logged out) + AUTH */}
          {(variant === "auth" || (variant === "public" && !hasToken)) && (
            <>
              {variant === "public" ? (
                <Link to="/pricing" style={pillLinkGhost}>
                  Preise
                </Link>
              ) : null}

              <Link to="/login" style={pillLinkGhost}>
                Login
              </Link>

              <Link to="/register" style={pillCta}>
                Registrieren
              </Link>
            </>
          )}
        </nav>
      </div>
    </header>
  );
}

/* STYLES */

const pillLinkGhost: React.CSSProperties = {
  textDecoration: "none",
  color: theme.colors.textSecondary,
  fontWeight: 600,
  padding: "10px 16px",
  borderRadius: theme.radius.pill,
};

const pillCta: React.CSSProperties = {
  textDecoration: "none",
  color: theme.colors.bgDeep,
  background: theme.gradients.ctaPrimary,
  fontWeight: 700,
  padding: "10px 18px",
  borderRadius: theme.radius.pill,
  boxShadow: "0 10px 24px rgba(0, 0, 0, 0.35)",
};

export default Header;
