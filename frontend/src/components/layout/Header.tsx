import { Link, NavLink } from "react-router-dom";
import { LogOut } from "lucide-react";
import { motion } from "framer-motion";
import { theme } from "../ui/theme";

type HeaderProps = {
  variant?: "public" | "auth" | "app";
  onLogout?: () => void;
};

const appLinks = [
  { to: "/app/dashboard", label: "Dashboard" },
  { to: "/app/analyze", label: "Analyse" },
  { to: "/app/custom-analysis", label: "Eigene Analyse" },
  { to: "/app/billing", label: "Billing" },
  { to: "/app/account", label: "Account" },
];

function Header({ variant = "public", onLogout }: HeaderProps) {
  const isApp = variant === "app";
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
          to={isApp || hasToken ? "/app/dashboard" : "/landing"}
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
            background: isApp ? theme.colors.panelAlt : "transparent",
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

          {/* APP */}
          {isApp &&
            appLinks.map((link) => (
              <NavLink key={link.to} to={link.to} style={navLinkStyle}>
                {({ isActive }) =>
                  isActive ? (
                    <motion.span
                      layoutId="header-active-pill"
                      style={pillActiveBg}
                      transition={theme.motion.spring}
                    >
                      <span style={{ position: "relative", zIndex: 1 }}>{link.label}</span>
                    </motion.span>
                  ) : (
                    <span>{link.label}</span>
                  )
                }
              </NavLink>
            ))}

          {isApp && (
            <motion.button
              onClick={onLogout}
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              style={{
                marginLeft: "6px",
                display: "inline-flex",
                alignItems: "center",
                gap: "6px",
                background: theme.colors.dangerSoft,
                color: theme.colors.dangerText,
                border: `1px solid ${theme.colors.dangerBorder}`,
                borderRadius: theme.radius.pill,
                padding: "8px 14px",
                fontWeight: 700,
                fontSize: "0.9rem",
                cursor: "pointer",
              }}
            >
              <LogOut size={15} />
              Logout
            </motion.button>
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

const navLinkStyle = ({ isActive }: { isActive: boolean }): React.CSSProperties => ({
  position: "relative",
  textDecoration: "none",
  color: isActive ? "#ffffff" : theme.colors.textSecondary,
  fontWeight: 600,
  fontSize: "0.92rem",
  padding: "9px 16px",
  borderRadius: theme.radius.pill,
  display: "inline-flex",
  alignItems: "center",
});

const pillActiveBg: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  borderRadius: theme.radius.pill,
  background: theme.gradients.ctaPrimary,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

export default Header;
