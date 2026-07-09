import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { theme } from "../ui/theme";
import Button from "../ui/Button";
import { loadPlausibleScript } from "../../lib/plausible";

const CONSENT_KEY = "analytics_consent";

/** Fixiertes Banner am unteren Bildschirmrand: fragt einmalig, ob Plausible
 * Analytics geladen werden darf. Plausible selbst nutzt keine Cookies und
 * keine personenbezogenen Daten (anonymisierte Seitenaufrufe/Referrer) -
 * das Banner bleibt trotzdem als allgemeines Consent-Framework bestehen,
 * falls spaeter echte First-Party-Cookies dazukommen. */
function CookieConsentBanner() {
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const consent = localStorage.getItem(CONSENT_KEY);
    if (consent === "accepted") {
      loadPlausibleScript();
    } else if (consent !== "rejected") {
      setIsVisible(true);
    }
  }, []);

  function handleAccept() {
    localStorage.setItem(CONSENT_KEY, "accepted");
    loadPlausibleScript();
    setIsVisible(false);
  }

  function handleReject() {
    localStorage.setItem(CONSENT_KEY, "rejected");
    setIsVisible(false);
  }

  return (
    <AnimatePresence>
      {isVisible ? (
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 40 }}
          transition={{ duration: 0.3 }}
          style={{
            position: "fixed",
            bottom: "20px",
            left: "20px",
            right: "20px",
            maxWidth: "560px",
            zIndex: 10000,
            display: "flex",
            flexDirection: "column",
            gap: "14px",
            padding: "22px 24px",
            borderRadius: theme.radius.lg,
            background: theme.glass.elevated.background,
            border: `1px solid ${theme.glass.elevated.border}`,
            boxShadow: theme.glass.elevated.shadow,
            backdropFilter: `blur(${theme.glass.elevated.blur})`,
            color: theme.colors.textPrimary,
          }}
        >
          <div style={{ fontSize: "0.98rem", lineHeight: 1.7, color: theme.colors.textSecondary }}>
            Wir nutzen{" "}
            <strong style={{ color: theme.colors.textPrimary }}>Plausible Analytics</strong> —
            datenschutzfreundlich, ohne Cookies und ohne personenbezogene
            Daten — um zu verstehen, wie ComAnalysis genutzt wird. Details
            dazu findest du in unseren{" "}
            <Link
              to="/legal/cookies"
              style={{ color: theme.colors.chrome, fontWeight: 700 }}
            >
              Cookie-Hinweisen
            </Link>
            .
          </div>

          <div style={{ display: "flex", gap: "12px", flexWrap: "wrap" }}>
            <Button variant="cta" onClick={handleAccept} style={{ flex: "1 1 auto" }}>
              Zustimmen
            </Button>
            <Button variant="ghost" onClick={handleReject} style={{ flex: "1 1 auto" }}>
              Ablehnen
            </Button>
          </div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

export default CookieConsentBanner;
