import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { theme } from "../ui/theme";
import Button from "../ui/Button";
import { loadWebAnalyticsScript } from "../../lib/webAnalytics";
import { useIsMobile } from "../../hooks/useMediaQuery";

const CONSENT_KEY = "analytics_consent";

/** Fixiertes Banner am unteren Bildschirmrand: fragt einmalig, ob Cloudflare
 * Web Analytics geladen werden darf. Cloudflare Web Analytics selbst nutzt
 * keine Cookies und keine personenbezogenen Daten (anonymisierte
 * Seitenaufrufe/Referrer) - das Banner bleibt trotzdem als allgemeines
 * Consent-Framework bestehen, falls spaeter echte First-Party-Cookies
 * dazukommen. */
function CookieConsentBanner() {
  const [isVisible, setIsVisible] = useState(false);
  const isMobile = useIsMobile();

  useEffect(() => {
    const consent = localStorage.getItem(CONSENT_KEY);
    if (consent === "accepted") {
      loadWebAnalyticsScript();
    } else if (consent !== "rejected") {
      // Kurze Verzögerung statt sofortiger Einblendung: Web Analytics lädt
      // ohnehin erst nach explizitem "Zustimmen" (siehe unten), eine
      // verzögerte Anzeige der reinen Abfrage ist DSGVO-unkritisch. Lässt
      // dem wichtigsten Conversion-Moment - der erste Blick auf den
      // Hero-CTA - einen Moment ungestört, bevor der Banner erscheint
      // (LAUNCH.md P2-14).
      const timer = setTimeout(() => setIsVisible(true), 1200);
      return () => clearTimeout(timer);
    }
  }, []);

  function handleAccept() {
    localStorage.setItem(CONSENT_KEY, "accepted");
    loadWebAnalyticsScript();
    setIsVisible(false);
  }

  function handleReject() {
    localStorage.setItem(CONSENT_KEY, "rejected");
    setIsVisible(false);
  }

  return (
    <AnimatePresence>
      {isVisible ? (
        // Schmale, volle Breite Bottom-Bar statt einer bis zu ~180px hohen
        // Karte unten links - die Karte überlappte beim Erstbesuch den
        // "Kostenlos starten"-CTA der Hero-Section (LAUNCH.md P2-14). Eine
        // niedrige, über die volle Breite gehende Leiste lässt am oberen
        // Viewport-Rand (wo Hero-CTAs typischerweise sitzen) deutlich mehr
        // Abstand als eine 20px-vom-Rand positionierte Karte.
        <motion.div
          initial={{ opacity: 0, y: 60 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 60 }}
          transition={{ duration: 0.3 }}
          style={{
            position: "fixed",
            bottom: 0,
            left: 0,
            right: 0,
            zIndex: 10000,
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: isMobile ? "8px 12px" : "12px 20px",
            // Zusätzlicher Bottom-Puffer für den iOS-Home-Indicator (0 auf
            // Geräten ohne Safe-Area-Inset) — siehe RESPONSIVE.md R-P0-5.
            padding: isMobile
              ? "10px 14px calc(10px + env(safe-area-inset-bottom))"
              : "14px 24px calc(14px + env(safe-area-inset-bottom))",
            background: theme.glass.elevated.background,
            borderTop: `1px solid ${theme.glass.elevated.border}`,
            boxShadow: theme.glass.elevated.shadow,
            backdropFilter: `blur(${theme.glass.elevated.blur})`,
            color: theme.colors.textPrimary,
          }}
        >
          <div
            style={{
              flex: "1 1 240px",
              fontSize: isMobile ? "0.78rem" : "0.9rem",
              lineHeight: 1.4,
              color: theme.colors.textSecondary,
            }}
          >
            {isMobile ? (
              <>
                <strong style={{ color: theme.colors.textPrimary }}>Cloudflare Web Analytics</strong>{" "}
                — cookielos, keine PII.{" "}
                <Link to="/legal/cookies" style={{ color: theme.colors.chrome, fontWeight: 700 }}>
                  Details
                </Link>
              </>
            ) : (
              <>
                Wir nutzen{" "}
                <strong style={{ color: theme.colors.textPrimary }}>Cloudflare Web Analytics</strong> —
                datenschutzfreundlich, ohne Cookies und ohne personenbezogene
                Daten. Details in unseren{" "}
                <Link to="/legal/cookies" style={{ color: theme.colors.chrome, fontWeight: 700 }}>
                  Cookie-Hinweisen
                </Link>
                .
              </>
            )}
          </div>

          <div style={{ display: "flex", gap: isMobile ? "8px" : "10px", flexShrink: 0 }}>
            <Button
              variant="ghost"
              onClick={handleReject}
              style={isMobile ? { padding: "7px 14px", fontSize: "0.8rem" } : undefined}
            >
              Ablehnen
            </Button>
            <Button
              variant="cta"
              onClick={handleAccept}
              style={isMobile ? { padding: "7px 14px", fontSize: "0.8rem" } : undefined}
            >
              Zustimmen
            </Button>
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

export default CookieConsentBanner;
