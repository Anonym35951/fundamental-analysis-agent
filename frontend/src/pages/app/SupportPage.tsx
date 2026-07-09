import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getCurrentUser } from "../../api/auth";
import { theme } from "../../components/ui/theme";
import SupportForm from "../../components/support/SupportForm";

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0 },
};

function SupportPage() {
  const [userEmail, setUserEmail] = useState<string | undefined>(undefined);

  useEffect(() => {
    let isMounted = true;
    getCurrentUser()
      .then((user) => {
        if (isMounted) setUserEmail(user.email);
      })
      .catch(() => {
        // Kein Blocker fürs Formular — E-Mail-Feld bleibt dann einfach leer.
      });
    return () => {
      isMounted = false;
    };
  }, []);

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={fadeUp}
      transition={{ duration: 0.4 }}
      style={pageWrapper}
    >
      <div style={pageHeader}>
        <div style={sectionEyebrow}>Hilfe & Kontakt</div>
        <h1 style={pageTitle}>Support</h1>
        <p style={pageSubtitle}>
          Frage zu deinem Konto, einer Analyse oder deinem Abo? Schreib uns —
          wir antworten in der Regel innerhalb von 1–2 Werktagen.
        </p>
      </div>

      <div style={formPanel}>
        <SupportForm prefilledEmail={userEmail} />
      </div>
    </motion.div>
  );
}

const pageWrapper = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "24px",
  padding: "10px 6px 20px",
  maxWidth: "640px",
};

const pageHeader = {
  marginBottom: "4px",
};

const sectionEyebrow = {
  fontSize: "0.82rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.06em",
  textTransform: "uppercase" as const,
  marginBottom: "8px",
};

const pageTitle = {
  fontSize: "2rem",
  fontWeight: 800,
  margin: "0 0 8px 0",
  color: theme.colors.textPrimary,
};

const pageSubtitle = {
  margin: 0,
  color: theme.colors.textSecondary,
  fontSize: "1rem",
  lineHeight: 1.6,
};

const formPanel = {
  padding: "28px 30px",
  borderRadius: theme.radius.lg,
  border: `1px solid ${theme.glass.subtle.border}`,
  background: theme.glass.subtle.background,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
};

export default SupportPage;
