import { Outlet } from "react-router-dom";
import Header from "../components/layout/Header";
import Footer from "../components/layout/Footer";
import ErrorBoundary from "../components/common/ErrorBoundary";
import AmbientBackground from "../components/ui/AmbientBackground";
import { theme } from "../components/ui/theme";
import { useScrollProgress } from "../hooks/useScrollProgress";

function PublicLayout() {
  useScrollProgress();
  return (
    <div
      style={{
        position: "relative",
        minHeight: "100dvh",
        display: "flex",
        flexDirection: "column",
        background: `${theme.gradients.heroBg}, linear-gradient(180deg, ${theme.colors.bgDeepAlt} 0%, ${theme.colors.bgDeep} 100%)`,
        color: theme.colors.textPrimary,
      }}
    >
      <AmbientBackground variant="hero" />

      {/* ✅ Einheitlicher Header */}
      <div style={{ position: "relative", zIndex: 1 }}>
        <Header variant="public" />
      </div>

      {/* Main Content */}
      <main
        style={{
          position: "relative",
          zIndex: 1,
          flex: 1,
          width: "100%",
        }}
      >
        <ErrorBoundary>
          <Outlet />
        </ErrorBoundary>
      </main>

      {/* ✅ Einheitlicher Footer */}
      <div style={{ position: "relative", zIndex: 1 }}>
        <Footer />
      </div>
    </div>
  );
}

export default PublicLayout;