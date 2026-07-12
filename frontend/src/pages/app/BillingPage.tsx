import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Check, Crown } from "lucide-react";
import { getCurrentUser } from "../../api/auth";
import { createCheckoutSession } from "../../api/billing";
import { theme } from "../../components/ui/theme";
import Modal from "../../components/ui/Modal";
import ParticleBeamBackground from "../../components/landing/ParticleBeamBackground";
import { FREE_PLAN, PRO_PLAN } from "../../config/pricingPlans";

type BillingInterval = "month" | "year";

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0 },
};

function BillingPage() {
  const [currentPlan, setCurrentPlan] = useState("free");
  const [isLoadingUser, setIsLoadingUser] = useState(true);
  const [isStartingCheckout, setIsStartingCheckout] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [billingInterval, setBillingInterval] =
    useState<BillingInterval>("year");

  // Widerrufs-Zustimmung (EU-Verbraucherrecht): vor dem Stripe-Checkout muss
  // der Nutzer der sofortigen Leistungserbringung zustimmen und das damit
  // verbundene Erlöschen des Widerrufsrechts bestätigen.
  const [isConsentModalOpen, setIsConsentModalOpen] = useState(false);
  const [hasConsented, setHasConsented] = useState(false);

  const isPro = currentPlan === "pro";
  const isYearly = billingInterval === "year";

  useEffect(() => {
    async function loadCurrentUser() {
      try {
        setErrorMessage("");
        const user = await getCurrentUser();
        setCurrentPlan(user.plan);
      } catch (error) {
        if (error instanceof Error) {
          setErrorMessage(error.message);
        } else {
          setErrorMessage("Benutzerdaten konnten nicht geladen werden.");
        }
      } finally {
        setIsLoadingUser(false);
      }
    }

    loadCurrentUser();
  }, []);

  function handleUpgradeClick() {
    if (isStartingCheckout || isPro) {
      return;
    }
    setHasConsented(false);
    setIsConsentModalOpen(true);
  }

  async function startCheckout() {
    if (isStartingCheckout || !hasConsented) {
      return;
    }

    try {
      setErrorMessage("");
      setIsStartingCheckout(true);

      const checkoutUrl = await createCheckoutSession(billingInterval);
      window.location.href = checkoutUrl;
    } catch (error) {
      setIsConsentModalOpen(false);
      if (error instanceof Error) {
        setErrorMessage(error.message);
      } else {
        setErrorMessage("Checkout konnte nicht gestartet werden.");
      }
      setIsStartingCheckout(false);
    }
  }

  const upgradeButtonLabel = isStartingCheckout
    ? "Checkout wird gestartet..."
    : isYearly
    ? "Jährliches Pro starten"
    : "Monatliches Pro starten";

  return (
    <div style={pageWrapper}>
      <ParticleBeamBackground densityMultiplier={1.3} />

      {errorMessage ? (
        <div
          style={{
            position: "relative",
            padding: "14px 16px",
            borderRadius: "14px",
            background: theme.colors.dangerSoft,
            border: `1px solid ${theme.colors.dangerBorder}`,
            color: theme.colors.dangerText,
            fontSize: "0.98rem",
            lineHeight: 1.7,
          }}
        >
          {errorMessage}
        </div>
      ) : null}

      {/* Hero */}
      <motion.section
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        transition={{ duration: 0.4 }}
        style={heroSection}
      >
        <div aria-hidden="true" style={heroHaloGlow} />
        <div aria-hidden="true" style={heroWatermark}>
          PRO
        </div>

        <div style={{ position: "relative", zIndex: 1 }}>
          <div style={heroTopRow}>
            <div style={heroBadge}>Upgrade & Abonnement</div>
            {!isLoadingUser ? (
              <div style={heroStatusChip}>
                Aktueller Tarif: {isPro ? "Pro" : "Free"}
              </div>
            ) : null}
          </div>

          <h1 style={heroTitle}>
            Mehr Tiefe, mehr Freiheit, mehr Analyse-Power
          </h1>

          <p style={heroText}>
            Schalte mit Pro einen deutlich leistungsfähigeren Analyse-Workflow
            frei. Erhalte mehr Spielraum für deine Auswertungen, nutze die App
            intensiver und arbeite strukturierter mit deinem persönlichen
            Investment-Prozess.
          </p>

          <div style={heroActionRow}>
            {isLoadingUser ? (
              <div style={loadingPill}>Tarif wird geladen...</div>
            ) : !isPro ? (
              <button
                type="button"
                onClick={handleUpgradeClick}
                disabled={isStartingCheckout}
                style={{
                  ...primaryButton,
                  border: "none",
                  cursor: isStartingCheckout ? "not-allowed" : "pointer",
                  opacity: isStartingCheckout ? 0.82 : 1,
                }}
              >
                {upgradeButtonLabel}
              </button>
            ) : (
              <div style={successPill}>Dein Pro-Plan ist aktiv</div>
            )}

            <Link to="/app/account" style={secondaryButton}>
              Konto verwalten
            </Link>
          </div>
        </div>
      </motion.section>

      {/* Interval toggle */}
      {!isPro ? (
        <motion.section
          initial="hidden"
          animate="visible"
          variants={fadeUp}
          transition={{ duration: 0.4, delay: 0.05 }}
          style={intervalToggleSection}
        >
          <div style={intervalTogglePill}>
            <button
              type="button"
              onClick={() => setBillingInterval("month")}
              disabled={isStartingCheckout}
              style={{
                ...intervalToggleOption,
                ...(billingInterval === "month"
                  ? intervalToggleOptionActive
                  : intervalToggleOptionInactive),
                cursor: isStartingCheckout ? "not-allowed" : "pointer",
              }}
            >
              Monatlich
            </button>
            <button
              type="button"
              onClick={() => setBillingInterval("year")}
              disabled={isStartingCheckout}
              style={{
                ...intervalToggleOption,
                ...(billingInterval === "year"
                  ? intervalToggleOptionActive
                  : intervalToggleOptionInactive),
                cursor: isStartingCheckout ? "not-allowed" : "pointer",
              }}
            >
              Jährlich
            </button>
          </div>

          <div style={intervalSavingsBadge}>2 Monate gratis</div>

          <div style={intervalHint}>
            {isYearly
              ? "500 € / Jahr statt 600 € — entspricht 41,67 € pro Monat."
              : "50 € / Monat — flexibel und jederzeit kündbar."}
          </div>
        </motion.section>
      ) : null}

      {/* Pricing */}
      <motion.section
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        transition={{ duration: 0.4, delay: 0.1 }}
        style={pricingSection}
      >
        <div style={sectionHeader}>
          <div style={sectionEyebrow}>Tarifvergleich</div>
          <h2 style={sectionTitle}>Free vs. Pro</h2>
          <p style={sectionText}>
            Wähle den Plan, der zu deiner aktuellen Nutzung passt. Du kannst
            mit Free starten und später jederzeit auf Pro upgraden.
          </p>
        </div>

        <div style={pricingGrid} data-tour="billing-plans">
          <div style={freeCard}>
            <div style={planNameRow}>
              <div style={planName}>{FREE_PLAN.name}</div>
              <div style={planTagMuted}>{FREE_PLAN.tag}</div>
            </div>

            <div style={planPrice}>{FREE_PLAN.priceLabel}</div>

            <p style={planDescription}>{FREE_PLAN.description}</p>

            <div style={featureList}>
              {FREE_PLAN.features.map((item) => (
                <div key={item} style={featureItem}>
                  <Check
                    size={15}
                    color={theme.colors.success}
                    style={{ flexShrink: 0 }}
                  />
                  <span>{item}</span>
                </div>
              ))}
            </div>

            <div style={freeFooter}>
              {isLoadingUser
                ? "Tarif wird geladen"
                : isPro
                ? "Du hast bereits einen höheren Plan."
                : "Aktueller Startpunkt"}
            </div>
          </div>

          <div style={proCard}>
            <div style={planNameRow}>
              <div
                style={{
                  ...planName,
                  color: "#ffffff",
                  display: "flex",
                  alignItems: "center",
                  gap: "8px",
                }}
              >
                <Crown size={20} color="#ffffff" />
                {PRO_PLAN.name}
              </div>
              <div
                style={{
                  ...planTag,
                  background: "rgba(212, 212, 216, 0.16)",
                  border: "1px solid rgba(212, 212, 216, 0.3)",
                  color: "#ffffff",
                }}
              >
                {PRO_PLAN.tag}
              </div>
            </div>

            <div style={proPriceBlock}>
              {isYearly ? (
                <div style={proYearlyPriceWrap}>
                  <span style={proYearlyOldPrice}>{PRO_PLAN.yearlyOldPriceLabel}</span>
                  <span style={proMainPrice}>{PRO_PLAN.yearlyPriceLabel}</span>
                </div>
              ) : (
                <div style={proMainPrice}>{PRO_PLAN.monthlyPriceLabel}</div>
              )}

              {isYearly ? (
                <div style={proSavingsRow}>
                  <span style={proSavingsBadge}>{PRO_PLAN.savingsBadge}</span>
                  <span style={proSavingsText}>{PRO_PLAN.savingsText}</span>
                </div>
              ) : (
                <div style={proMonthlyHint}>{PRO_PLAN.monthlyHint}</div>
              )}
            </div>

            <p style={planDescriptionBright}>{PRO_PLAN.description}</p>

            <div style={featureList}>
              {PRO_PLAN.features.map((item) => (
                <div key={item} style={featureItemBright}>
                  <Check
                    size={15}
                    color="#ffffff"
                    style={{ flexShrink: 0 }}
                  />
                  <span>{item}</span>
                </div>
              ))}
            </div>

            {!isLoadingUser && !isPro ? (
              <div style={proSelectionNote}>
                Ausgewählt: {isYearly ? PRO_PLAN.yearlySelectionNote : "Monatlich"}
              </div>
            ) : null}

            {isLoadingUser ? (
              <div style={proActiveBadge}>Tarif wird geladen...</div>
            ) : !isPro ? (
              <button
                type="button"
                onClick={handleUpgradeClick}
                disabled={isStartingCheckout}
                style={{
                  ...proButton,
                  border: "none",
                  cursor: isStartingCheckout ? "not-allowed" : "pointer",
                  opacity: isStartingCheckout ? 0.82 : 1,
                }}
              >
                {isStartingCheckout
                  ? "Wird gestartet..."
                  : isYearly
                  ? "Zu Pro jährlich wechseln"
                  : "Zu Pro monatlich wechseln"}
              </button>
            ) : (
              <div style={proActiveBadge}>Pro ist aktiv</div>
            )}
          </div>
        </div>
      </motion.section>

      {/* Benefits */}
      <motion.section
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        transition={{ duration: 0.4, delay: 0.15 }}
        style={benefitsSection}
      >
        <div style={sectionHeader}>
          <div style={sectionEyebrow}>Nutzen von Pro</div>
          <h2 style={sectionTitle}>Warum sich das Upgrade lohnt</h2>
          <p style={sectionText}>
            Pro soll sich nicht nur nach mehr Funktionen anfühlen, sondern
            nach einem klar besseren Produkt.
          </p>
        </div>

        <div style={benefitsGrid}>
          {[
            {
              title: "Mehr Fokus",
              text: "Arbeite mit weniger Reibung und nutze die Plattform intensiver für echte Analysearbeit.",
            },
            {
              title: "Mehr Produktivität",
              text: "Führe deine Investment-Analysen schneller und sauberer in einer professionelleren Umgebung durch.",
            },
            {
              title: "Mehr Anspruch",
              text: "Pro richtet sich an Nutzer, die ihr Setup ausbauen und ComAnalysis stärker in ihren Workflow integrieren wollen.",
            },
          ].map((item) => (
            <motion.div
              key={item.title}
              style={benefitCard}
              whileHover={{ y: -5, scale: 1.01 }}
              transition={theme.motion.spring}
            >
              <div style={benefitTitle}>{item.title}</div>
              <div style={benefitText}>{item.text}</div>
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Final CTA */}
      <motion.section
        initial="hidden"
        animate="visible"
        variants={fadeUp}
        transition={{ duration: 0.4, delay: 0.2 }}
        style={finalCtaSection}
      >
        <div style={finalCtaEyebrow}>Bereit für den nächsten Schritt?</div>
        <h2 style={finalCtaTitle}>Hol mehr aus ComAnalysis heraus</h2>
        <p style={finalCtaText}>
          Upgrade auf Pro und nutze die Plattform mit mehr Freiheit, mehr
          Tiefe und mehr professioneller Substanz.
        </p>

        {isLoadingUser ? (
          <div style={proAlreadyActive}>Tarif wird geladen...</div>
        ) : !isPro ? (
          <div style={finalCtaActionRow}>
            <button
              type="button"
              onClick={handleUpgradeClick}
              disabled={isStartingCheckout}
              style={{
                ...finalPrimaryButton,
                border: "none",
                cursor: isStartingCheckout ? "not-allowed" : "pointer",
                opacity: isStartingCheckout ? 0.82 : 1,
              }}
            >
              {upgradeButtonLabel}
            </button>

            <Link to="/app/dashboard" style={finalSecondaryButton}>
              Zurück zum Dashboard
            </Link>
          </div>
        ) : (
          <div style={proAlreadyActive}>Dein Pro-Zugang ist bereits aktiv.</div>
        )}
      </motion.section>

      <Modal
        isOpen={isConsentModalOpen}
        onClose={() => {
          if (!isStartingCheckout) setIsConsentModalOpen(false);
        }}
        title="Bevor es losgeht"
      >
        <p
          style={{
            margin: "0 0 14px 0",
            color: theme.colors.textSecondary,
            fontSize: "0.98rem",
            lineHeight: 1.7,
          }}
        >
          Du schließt ein {isYearly ? "jährliches" : "monatliches"} Pro-Abonnement
          ab. Die Zahlung wird sicher über Stripe abgewickelt.
        </p>

        <label
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: "10px",
            padding: "12px 14px",
            borderRadius: theme.radius.md,
            border: `1px solid ${theme.glass.subtle.border}`,
            background: theme.glass.subtle.background,
            color: theme.colors.textSecondary,
            fontSize: "0.92rem",
            lineHeight: 1.6,
            cursor: "pointer",
          }}
        >
          <input
            type="checkbox"
            checked={hasConsented}
            onChange={(event) => setHasConsented(event.target.checked)}
            disabled={isStartingCheckout}
            style={{ marginTop: "3px", flexShrink: 0 }}
          />
          <span>
            Ich stimme ausdrücklich zu, dass ComAnalysis vor Ablauf der
            Widerrufsfrist mit der Ausführung des Vertrags beginnt. Mir ist
            bekannt, dass mein Widerrufsrecht mit Beginn der Vertragsausführung
            erlischt. Die{" "}
            <Link
              to="/legal/terms"
              target="_blank"
              style={{ color: theme.colors.chrome, fontWeight: 700 }}
            >
              Nutzungsbedingungen
            </Link>{" "}
            habe ich zur Kenntnis genommen.
          </span>
        </label>

        <button
          type="button"
          onClick={startCheckout}
          disabled={!hasConsented || isStartingCheckout}
          style={{
            marginTop: "16px",
            width: "100%",
            padding: "14px 18px",
            borderRadius: theme.radius.md,
            border: "none",
            background: theme.colors.chrome,
            color: theme.colors.black,
            fontWeight: 800,
            fontSize: "1rem",
            cursor:
              !hasConsented || isStartingCheckout ? "not-allowed" : "pointer",
            opacity: !hasConsented || isStartingCheckout ? 0.55 : 1,
          }}
        >
          {isStartingCheckout ? "Checkout wird gestartet..." : "Weiter zu Stripe"}
        </button>
      </Modal>
    </div>
  );
}

/* styles */

const pageWrapper = {
  position: "relative" as const,
  display: "flex",
  flexDirection: "column" as const,
  gap: "34px",
  padding: "10px 6px 20px",
};

const heroSection = {
  position: "relative" as const,
  background: theme.glass.elevated.background,
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  borderRadius: "28px",
  padding: "34px 34px 36px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
  overflow: "hidden" as const,
};

/** Soft white halo glow bleeding from behind the hero headline. */
const heroHaloGlow = {
  position: "absolute" as const,
  top: "-30%",
  left: "50%",
  transform: "translateX(-50%)",
  width: "70%",
  height: "140%",
  background: "radial-gradient(circle, rgba(255,255,255,0.12), transparent 65%)",
  filter: "blur(10px)",
  pointerEvents: "none" as const,
  zIndex: 0,
};

/** Oversized faint watermark word for visual depth, inspired by the
 * reference pricing-page layout. */
const heroWatermark = {
  position: "absolute" as const,
  top: "-6%",
  right: "-2%",
  fontSize: "clamp(6rem, 16vw, 13rem)",
  fontWeight: 900,
  color: "rgba(255, 255, 255, 0.04)",
  letterSpacing: "-0.04em",
  lineHeight: 1,
  pointerEvents: "none" as const,
  zIndex: 0,
  userSelect: "none" as const,
};

const heroTopRow = {
  display: "flex",
  alignItems: "center",
  gap: "12px",
  flexWrap: "wrap" as const,
  marginBottom: "16px",
};

const heroBadge = {
  display: "inline-block",
  padding: "8px 12px",
  borderRadius: "999px",
  background: theme.colors.panelAlt,
  border: "1px solid rgba(96, 165, 250, 0.16)",
  color: theme.colors.chrome,
  fontSize: "0.86rem",
  fontWeight: 700,
  letterSpacing: "0.03em",
};

const heroStatusChip = {
  display: "inline-block",
  padding: "8px 12px",
  borderRadius: "999px",
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.border}`,
  color: theme.colors.textPrimary,
  fontSize: "0.86rem",
  fontWeight: 700,
};

const heroTitle = {
  margin: "0 0 16px 0",
  fontSize: "3rem",
  lineHeight: 1.05,
  letterSpacing: "-0.045em",
  background: theme.gradients.ctaPrimary,
  WebkitBackgroundClip: "text" as const,
  backgroundClip: "text" as const,
  color: "transparent",
};

const heroText = {
  margin: 0,
  maxWidth: "860px",
  color: theme.colors.textPrimary,
  fontSize: "1.18rem",
  lineHeight: 1.9,
};

const heroActionRow = {
  display: "flex",
  gap: "14px",
  flexWrap: "wrap" as const,
  marginTop: "26px",
};

const primaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "999px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  fontWeight: 800,
  fontSize: "1rem",
  boxShadow: "0 14px 30px rgba(0, 0, 0, 0.35)",
};

const secondaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "999px",
  background: theme.colors.panel,
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "1rem",
  border: "1px solid rgba(148, 163, 184, 0.14)",
};

const successPill = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "999px",
  background: "rgba(22, 163, 74, 0.16)",
  color: theme.colors.successText,
  fontWeight: 800,
  fontSize: "1rem",
  border: "1px solid rgba(134, 239, 172, 0.24)",
};

const loadingPill = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "999px",
  background: "rgba(255,255,255,0.08)",
  color: theme.colors.textSecondary,
  fontWeight: 800,
  fontSize: "1rem",
  border: "1px solid rgba(148, 163, 184, 0.18)",
};

const intervalToggleSection = {
  display: "flex",
  alignItems: "center",
  gap: "16px",
  flexWrap: "wrap" as const,
};

const intervalTogglePill = {
  display: "inline-flex",
  alignItems: "center",
  gap: "6px",
  padding: "6px",
  borderRadius: "999px",
  background: theme.colors.panel,
  border: `1px solid ${theme.colors.border}`,
};

const intervalToggleOption = {
  padding: "11px 20px",
  borderRadius: "999px",
  fontWeight: 800,
  fontSize: "0.94rem",
  border: "none",
  transition: "all 0.18s ease",
};

const intervalToggleOptionActive = {
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  boxShadow: "0 10px 22px rgba(0, 0, 0, 0.3)",
};

const intervalToggleOptionInactive = {
  background: "transparent",
  color: theme.colors.textSecondary,
};

const intervalSavingsBadge = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "8px 12px",
  borderRadius: "999px",
  background: theme.colors.successSoft,
  border: `1px solid ${theme.colors.successBorder}`,
  color: theme.colors.successText,
  fontSize: "0.84rem",
  fontWeight: 800,
};

const intervalHint = {
  color: theme.colors.textSecondary,
  fontSize: "0.92rem",
};

const pricingSection = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "22px",
};

const sectionHeader = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "8px",
};

const sectionEyebrow = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const sectionTitle = {
  margin: 0,
  fontSize: "2rem",
  lineHeight: 1.15,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.03em",
};

const sectionText = {
  margin: 0,
  color: theme.colors.textPrimary,
  fontSize: "1.04rem",
  lineHeight: 1.8,
  maxWidth: "760px",
};

const pricingGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(min(320px, 100%), 1fr))",
  gap: "22px",
};

const freeCard = {
  background: theme.glass.elevated.background,
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  borderRadius: "24px",
  padding: "28px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: "0 14px 34px rgba(0, 0, 0, 0.22)",
  display: "flex",
  flexDirection: "column" as const,
};

const proCard = {
  background: "linear-gradient(135deg, #1c1c1f 0%, #2a2a2e 60%, #3a3a3f 100%)",
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  borderRadius: "24px",
  padding: "28px",
  border: `1px solid ${theme.colors.chromeBorder}`,
  boxShadow: theme.glass.elevated.shadow,
  display: "flex",
  flexDirection: "column" as const,
};

const planNameRow = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: "18px",
  gap: "12px",
};

const planName = {
  fontSize: "1.5rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
};

const planTag = {
  padding: "7px 10px",
  borderRadius: "999px",
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "0.82rem",
};

const planTagMuted = {
  padding: "7px 10px",
  borderRadius: "999px",
  background: theme.colors.panelAlt,
  color: theme.colors.textSecondary,
  fontWeight: 700,
  fontSize: "0.82rem",
};

const planPrice = {
  fontSize: "2rem",
  fontWeight: 900,
  color: theme.colors.textPrimary,
  marginBottom: "10px",
};

const proPriceBlock = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "10px",
  marginBottom: "10px",
};

const proMainPrice = {
  fontSize: "2rem",
  fontWeight: 900,
  color: "#ffffff",
  lineHeight: 1.1,
};

const proYearlyPriceWrap = {
  display: "flex",
  alignItems: "baseline",
  gap: "10px",
  flexWrap: "wrap" as const,
};

const proYearlyOldPrice = {
  color: "rgba(219, 234, 254, 0.78)",
  fontSize: "1.2rem",
  fontWeight: 800,
  textDecoration: "line-through",
};

const proSavingsRow = {
  display: "flex",
  alignItems: "center",
  gap: "10px",
  flexWrap: "wrap" as const,
};

const proSavingsBadge = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "7px 10px",
  borderRadius: "999px",
  background: "rgba(22, 163, 74, 0.18)",
  color: theme.colors.successText,
  fontSize: "0.8rem",
  fontWeight: 800,
  border: "1px solid rgba(134, 239, 172, 0.24)",
};

const proSavingsText = {
  color: "rgba(255, 255, 255, 0.85)",
  fontSize: "0.92rem",
  fontWeight: 700,
};

const proMonthlyHint = {
  color: "rgba(255, 255, 255, 0.85)",
  fontSize: "0.92rem",
  fontWeight: 700,
};

const planDescription = {
  margin: "0 0 18px 0",
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  lineHeight: 1.75,
};

const planDescriptionBright = {
  margin: "0 0 18px 0",
  color: "rgba(255, 255, 255, 0.85)",
  fontSize: "1rem",
  lineHeight: 1.75,
};

const featureList = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "10px",
  marginBottom: "22px",
};

const featureItem = {
  display: "flex",
  gap: "10px",
  alignItems: "flex-start",
  color: theme.colors.textSecondary,
  lineHeight: 1.65,
};

const featureItemBright = {
  display: "flex",
  gap: "10px",
  alignItems: "flex-start",
  color: "rgba(255, 255, 255, 0.85)",
  lineHeight: 1.65,
};

const freeFooter = {
  marginTop: "auto",
  color: theme.colors.textMuted,
  fontWeight: 600,
};

const proButton = {
  marginTop: "auto",
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 16px",
  borderRadius: "14px",
  background: "#ffffff",
  color: "#0c0c0e",
  fontWeight: 800,
  fontSize: "1rem",
};

const proActiveBadge = {
  marginTop: "auto",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 16px",
  borderRadius: "14px",
  background: "rgba(255,255,255,0.14)",
  color: "#ffffff",
  fontWeight: 800,
  fontSize: "1rem",
};

const proSelectionNote = {
  marginBottom: "14px",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "10px 12px",
  borderRadius: "12px",
  background: "rgba(255,255,255,0.12)",
  color: "#ffffff",
  fontWeight: 700,
  fontSize: "0.92rem",
};

const benefitsSection = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "22px",
};

const benefitsGrid = {
  display: "grid",
  // auto-fit statt fixem repeat(3, ...): kollabiert auf schmalen Screens von
  // selbst auf 1 Spalte, statt drei ~105px-Spalten auf 375px zu quetschen
  // (RESPONSIVE.md R-P0-7).
  gridTemplateColumns: "repeat(auto-fit, minmax(min(240px, 100%), 1fr))",
  gap: "20px",
};

const benefitCard = {
  background: theme.colors.panel,
  borderRadius: "20px",
  padding: "24px",
  border: "1px solid rgba(148, 163, 184, 0.14)",
  color: theme.colors.textPrimary,
  boxShadow: "0 16px 36px rgba(0, 0, 0, 0.24)",
};

const benefitTitle = {
  fontSize: "1.24rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
  marginBottom: "10px",
};

const benefitText = {
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  lineHeight: 1.75,
};

const finalCtaSection = {
  background: theme.colors.panel,
  borderRadius: "28px",
  padding: "34px",
  border: "1px solid rgba(148, 163, 184, 0.18)",
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.35)",
  textAlign: "center" as const,
};

const finalCtaEyebrow = {
  fontSize: "0.86rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
  marginBottom: "12px",
};

const finalCtaTitle = {
  margin: "0 0 14px 0",
  fontSize: "2.2rem",
  lineHeight: 1.15,
  color: theme.colors.textPrimary,
  letterSpacing: "-0.035em",
};

const finalCtaText = {
  margin: "0 auto",
  maxWidth: "780px",
  color: theme.colors.textPrimary,
  fontSize: "1.08rem",
  lineHeight: 1.85,
};

const finalCtaActionRow = {
  display: "flex",
  justifyContent: "center",
  gap: "14px",
  flexWrap: "wrap" as const,
  marginTop: "26px",
};

const finalPrimaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "15px 18px",
  borderRadius: "14px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  fontWeight: 800,
  fontSize: "1rem",
  boxShadow: "0 14px 30px rgba(0, 0, 0, 0.35)",
};

const finalSecondaryButton = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "15px 18px",
  borderRadius: "14px",
  background: theme.colors.panel,
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "1rem",
  border: "1px solid rgba(148, 163, 184, 0.14)",
};

const proAlreadyActive = {
  marginTop: "24px",
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  padding: "14px 18px",
  borderRadius: "14px",
  background: "rgba(22, 163, 74, 0.16)",
  color: theme.colors.successText,
  fontWeight: 800,
  fontSize: "1rem",
  border: "1px solid rgba(134, 239, 172, 0.24)",
};

export default BillingPage;
