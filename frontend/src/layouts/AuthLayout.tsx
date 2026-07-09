import { Outlet } from "react-router-dom";
import { motion } from "framer-motion";
import Header from "../components/layout/Header";
import Footer from "../components/layout/Footer";
import ErrorBoundary from "../components/common/ErrorBoundary";
import AmbientBackground from "../components/ui/AmbientBackground";
import ParticleBeamBackground from "../components/landing/ParticleBeamBackground";
import { theme } from "../components/ui/theme";
import { useScrollProgress } from "../hooks/useScrollProgress";

function AuthLayout() {
  useScrollProgress();
  return (
    <div
      style={{
        position: "relative",
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        background: `linear-gradient(180deg, ${theme.colors.bgDeepAlt} 0%, ${theme.colors.bgDeep} 100%)`,
        color: theme.colors.textPrimary,
      }}
    >
      <AmbientBackground variant="subtle" />
      <ParticleBeamBackground />

      <div style={{ position: "relative", zIndex: 1 }}>
        <Header variant="auth" />
      </div>

      {/* Main Content */}
      <main
        style={{
          position: "relative",
          zIndex: 1,
          flex: 1,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          padding: "56px 20px 56px",
        }}
      >
        <div
          style={{
            width: "100%",
            maxWidth: "560px",
          }}
        >
          {/* Top Text */}
          <div
            style={{
              marginBottom: "28px",
              textAlign: "center",
            }}
          >
            <div
              style={{
                display: "inline-block",
                padding: "9px 14px",
                borderRadius: theme.radius.pill,
                background: theme.colors.chromeSoft,
                color: theme.colors.chrome,
                fontWeight: 700,
                fontSize: "0.9rem",
                marginBottom: "16px",
                border: `1px solid ${theme.colors.chromeBorder}`,
              }}
            >
              Willkommen bei ComAnalysis
            </div>

            <h1
              style={{
                margin: "0 0 14px 0",
                fontSize: "2.2rem",
                lineHeight: 1.15,
                letterSpacing: "-0.035em",
                color: theme.colors.textPrimary,
              }}
            >
              Zugriff auf professionelle Fundamentalanalysen
            </h1>

            <p
              style={{
                margin: 0,
                color: theme.colors.textSecondary,
                fontSize: "1.08rem",
                lineHeight: 1.8,
              }}
            >
              Melde dich an oder registriere dich, um Analysen zu starten,
              eigene Methoden zu erstellen und dein Abo zu verwalten.
            </p>
          </div>

          {/* Card */}
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
            style={{
              background: theme.glass.elevated.background,
              backdropFilter: `blur(${theme.glass.elevated.blur})`,
              WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
              borderRadius: theme.radius.lg,
              padding: "30px",
              boxShadow: theme.glass.elevated.shadow,
              border: `1px solid ${theme.glass.elevated.border}`,
            }}
          >
            <ErrorBoundary>
              <Outlet />
            </ErrorBoundary>
          </motion.div>
        </div>
      </main>

      {/* ✅ Einheitlicher Footer */}
      <div style={{ position: "relative", zIndex: 1 }}>
        <Footer />
      </div>
    </div>
  );
}

export default AuthLayout;