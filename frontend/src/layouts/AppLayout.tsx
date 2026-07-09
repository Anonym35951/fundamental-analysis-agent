import { useEffect, useState } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Menu, X } from "lucide-react";
import Footer from "../components/layout/Footer";
import AppSidebar from "../components/layout/AppSidebar";
import EmailVerificationBanner from "../components/layout/EmailVerificationBanner";
import SessionTimeoutWatcher from "../components/layout/SessionTimeoutWatcher";
import ErrorBoundary from "../components/common/ErrorBoundary";
import AmbientBackground from "../components/ui/AmbientBackground";
import IntroOverlay, { type OverlayPhase } from "../components/intro/IntroOverlay";
import { FLY_OUT_DURATION } from "../components/intro/RingsScene";
import AppTour from "../components/onboarding/AppTour";
import { useIsMobile } from "../hooks/useMediaQuery";
import { useScrollProgress } from "../hooks/useScrollProgress";
import { TourStatusProvider } from "../hooks/TourStatusProvider";
import { theme } from "../components/ui/theme";
import { clearAuthAndWorkspaceState } from "../api/client";

function AppLayout() {
  const navigate = useNavigate();
  const isMobile = useIsMobile();
  useScrollProgress();
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState(
    () => localStorage.getItem("app_sidebar_collapsed") === "true"
  );
  // Read once on first mount of this layout (i.e. the first time the user
  // enters the /app/* subtree after login) — React Router keeps this layout
  // route mounted across child route changes, so this only runs once per
  // login. Deliberately a *pure* read with no side effect: StrictMode calls
  // lazy useState initializers twice in dev, so clearing the flag here would
  // wipe it before the real (second) call ever sees it. The actual removal
  // happens in the effect below, which is safe to double-invoke.
  const [showIntro] = useState(() => sessionStorage.getItem("show_intro") === "1");
  // No intro this load → content starts "done"/fully visible immediately,
  // no fade-in delay for ordinary navigation. Only the very first post-login
  // mount starts at "black" and progresses through the real sequence.
  const [introPhase, setIntroPhase] = useState<OverlayPhase>(showIntro ? "black" : "done");

  useEffect(() => {
    if (showIntro) sessionStorage.removeItem("show_intro");
  }, [showIntro]);

  useEffect(() => {
    // Auf Mobile bleibt die Sidebar als Off-Canvas-Drawer immer initial
    // geschlossen, unabhängig von der gespeicherten Desktop-Präferenz.
    if (isMobile) {
      setIsSidebarCollapsed(true);
    }
  }, [isMobile]);

  function handleLogout() {
    clearAuthAndWorkspaceState();
    navigate("/login");
  }

  function handleToggleSidebar() {
    setIsSidebarCollapsed((prev) => {
      const next = !prev;
      // Auf Mobile ist isCollapsed nur der Drawer-Öffnungsstatus und keine
      // Layout-Präferenz – darf die gespeicherte Desktop-Einstellung nicht
      // überschreiben.
      if (!isMobile) {
        localStorage.setItem("app_sidebar_collapsed", String(next));
      }
      return next;
    });
  }

  return (
    <TourStatusProvider>
    <div
      style={{
        position: "relative",
        minHeight: "100vh",
        display: "flex",
        background: `radial-gradient(circle at top, ${theme.colors.bgGradientStart}, ${theme.colors.bgGradientEnd})`,
        color: theme.colors.textSecondary,
      }}
    >
      {showIntro ? <IntroOverlay onPhaseChange={setIntroPhase} /> : null}

      <AppTour introDone={introPhase === "done"} />

      <SessionTimeoutWatcher />

      {/* Fades in exactly in sync with IntroOverlay fading out (same
          FLY_OUT_DURATION, starting together at the "out" phase) — a real
          crossfade instead of the already-fully-rendered page popping into
          view the instant the overlay disappears. A no-op (opacity 1 from
          the start) on ordinary navigation, since introPhase starts at
          "done" whenever there's no intro to play. */}
      <motion.div
        initial={{ opacity: introPhase === "done" ? 1 : 0, y: introPhase === "done" ? 0 : 14 }}
        animate={
          introPhase === "out" || introPhase === "done"
            ? { opacity: 1, y: 0 }
            : { opacity: 0, y: 14 }
        }
        transition={{ duration: FLY_OUT_DURATION, ease: "easeInOut" }}
        style={{ display: "flex", flex: 1, minWidth: 0 }}
      >
        <AmbientBackground variant="subtle" />

        <AppSidebar
          onLogout={handleLogout}
          isCollapsed={isSidebarCollapsed}
          onToggleCollapse={handleToggleSidebar}
        />

        {isMobile && !isSidebarCollapsed ? (
          <div
            onClick={handleToggleSidebar}
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0, 0, 0, 0.6)",
              zIndex: 55,
            }}
          />
        ) : null}

        {isMobile ? (
          <button
            type="button"
            onClick={handleToggleSidebar}
            aria-label={isSidebarCollapsed ? "Menü öffnen" : "Menü schließen"}
            style={{
              position: "fixed",
              top: "16px",
              left: "16px",
              zIndex: 65,
              width: "44px",
              height: "44px",
              borderRadius: "999px",
              border: "1px solid rgba(148, 163, 184, 0.18)",
              background: "rgba(15, 23, 42, 0.92)",
              color: theme.colors.textSecondary,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {isSidebarCollapsed ? <Menu size={20} /> : <X size={20} />}
          </button>
        ) : null}

        <div
          style={{
            flex: 1,
            minHeight: "100vh",
            display: "flex",
            flexDirection: "column",
            minWidth: 0,
          }}
        >
          <main
            style={{
              flex: 1,
              width: "100%",
              maxWidth: "1400px",
              margin: "0 auto",
              padding: isMobile ? "84px 18px 28px" : "40px 28px",
              boxSizing: "border-box",
            }}
          >
            <EmailVerificationBanner />
            <ErrorBoundary>
              <Outlet />
            </ErrorBoundary>
          </main>

          <Footer variant="app" />
        </div>
      </motion.div>
    </div>
    </TourStatusProvider>
  );
}

export default AppLayout;