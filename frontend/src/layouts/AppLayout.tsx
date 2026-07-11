import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { Outlet, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Menu, X } from "lucide-react";
import Footer from "../components/layout/Footer";
import AppSidebar from "../components/layout/AppSidebar";
import EmailVerificationBanner from "../components/layout/EmailVerificationBanner";
import SessionTimeoutWatcher from "../components/layout/SessionTimeoutWatcher";
import ErrorBoundary from "../components/common/ErrorBoundary";
import AmbientBackground from "../components/ui/AmbientBackground";
import type { OverlayPhase } from "../components/intro/IntroOverlay";
import { FLY_OUT_DURATION } from "../components/intro/introTiming";
import AppTour from "../components/onboarding/AppTour";

// Lazy statt statisch importiert (P2-15, LAUNCH.md): IntroOverlay zieht
// RingsScene (three.js/@react-three/fiber) nach - ohne Lazy-Loading wäre
// dieser Chunk Teil des Haupt-Bundles, obwohl showIntro auf den allermeisten
// Seitenaufrufen (jede Navigation außer dem ersten Post-Login-Mount) false
// ist und die Komponente nie gerendert wird.
const IntroOverlay = lazy(() => import("../components/intro/IntroOverlay"));
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
  // happens in the effect below, which is safe to double-invoke. Kept in a
  // ref (not state) since it only gates the one-time cleanup effect, not a
  // render decision.
  const hadIntroFlag = useRef(sessionStorage.getItem("show_intro") === "1");
  // P2-15 (LAUNCH.md): the 3D ring intro is skipped - not just reduced to a
  // static frame - under prefers-reduced-motion (same convention as
  // ParticleBeamBackground.tsx/StackedCards.tsx etc.) and on mobile, where
  // the extra three.js chunk download plus WebGL cost buys little on a
  // screen too small to appreciate the animation. isMobile above is already
  // resolved synchronously (useIsMobile reads matchMedia in its own useState
  // initializer), so it's safe to read here in the same render.
  const [showIntro] = useState(() => {
    if (!hadIntroFlag.current) return false;
    const prefersReducedMotion =
      window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
    return !prefersReducedMotion && !isMobile;
  });
  // No intro this load → content starts "done"/fully visible immediately,
  // no fade-in delay for ordinary navigation. Only the very first post-login
  // mount starts at "black" and progresses through the real sequence.
  const [introPhase, setIntroPhase] = useState<OverlayPhase>(showIntro ? "black" : "done");

  useEffect(() => {
    if (hadIntroFlag.current) sessionStorage.removeItem("show_intro");
  }, []);

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
      {showIntro ? (
        <Suspense fallback={null}>
          <IntroOverlay onPhaseChange={setIntroPhase} />
        </Suspense>
      ) : null}

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