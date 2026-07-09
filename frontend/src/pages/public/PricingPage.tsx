import { useState } from "react";
import type { CSSProperties } from "react";
import { Link } from "react-router-dom";
import { Check, Crown } from "lucide-react";
import SectionHeading from "../../components/landing/SectionHeading";
import { theme } from "../../components/ui/theme";
import { FREE_PLAN, PRO_PLAN } from "../../config/pricingPlans";

type BillingInterval = "month" | "year";

/** Öffentliche Preis-Seite — vor der Registrierung erreichbar (LAUNCH.md
 * P1-4). Nutzt dieselbe Plan-Datenquelle wie die authentifizierte
 * BillingPage (config/pricingPlans.ts), damit beide nie auseinanderlaufen.
 * CTAs führen zur Registrierung, nicht direkt zu Stripe — der eigentliche
 * Checkout ist erst nach Login möglich (siehe BillingPage). */
function PricingPage() {
  const [billingInterval, setBillingInterval] = useState<BillingInterval>("year");
  const isYearly = billingInterval === "year";

  return (
    <div style={pageWrapper}>
      <SectionHeading
        eyebrow="Preise"
        title="Ein fairer Einstieg, ein klarer Umstieg"
        subtitle="Starte kostenlos mit 50 Analyse-Einheiten im Monat. Wechsle zu Pro, sobald du regelmäßiger und ohne Limit analysieren willst."
      />

      <div style={toggleRow}>
        <button
          type="button"
          onClick={() => setBillingInterval("month")}
          style={toggleButton(!isYearly)}
        >
          Monatlich
        </button>
        <button
          type="button"
          onClick={() => setBillingInterval("year")}
          style={toggleButton(isYearly)}
        >
          Jährlich
        </button>
      </div>

      <div style={pricingGrid}>
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
                <Check size={15} color={theme.colors.success} style={{ flexShrink: 0 }} />
                <span>{item}</span>
              </div>
            ))}
          </div>

          <Link to="/register" style={freeCta}>
            Kostenlos starten
          </Link>
        </div>

        <div style={proCard}>
          <div style={planNameRow}>
            <div style={{ ...planName, color: "#ffffff", display: "flex", alignItems: "center", gap: "8px" }}>
              <Crown size={20} color="#ffffff" />
              {PRO_PLAN.name}
            </div>
            <div style={planTagBright}>{PRO_PLAN.tag}</div>
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
                <Check size={15} color="#ffffff" style={{ flexShrink: 0 }} />
                <span>{item}</span>
              </div>
            ))}
          </div>

          <Link to="/register" style={proCta}>
            Kostenlos registrieren
          </Link>
        </div>
      </div>

      <p style={disclaimerText}>
        Zahlungsabwicklung sicher über Stripe. Monatliche Abos sind jederzeit
        kündbar; das Upgrade findest du nach der Registrierung unter
        Account → Billing.
      </p>
    </div>
  );
}

export default PricingPage;

/* styles — bewusst eigenständig statt aus BillingPage.tsx importiert (dort
 * private Konstanten), aber inhaltlich an dieselbe Kartenoptik angelehnt. */

const pageWrapper: CSSProperties = {
  maxWidth: "1040px",
  margin: "0 auto",
  padding: "72px 24px 96px",
};

const toggleRow: CSSProperties = {
  display: "flex",
  justifyContent: "center",
  gap: "6px",
  padding: "5px",
  borderRadius: theme.radius.pill,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.border}`,
  width: "fit-content",
  margin: "0 auto 40px",
};

const toggleButton = (active: boolean): CSSProperties => ({
  padding: "9px 20px",
  borderRadius: theme.radius.pill,
  border: "none",
  cursor: "pointer",
  fontWeight: 700,
  fontSize: "0.9rem",
  background: active ? theme.gradients.ctaPrimary : "transparent",
  color: active ? theme.colors.bgDeep : theme.colors.textSecondary,
});

const pricingGrid: CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))",
  gap: "20px",
};

const cardBase: CSSProperties = {
  borderRadius: "24px",
  padding: "30px",
  display: "flex",
  flexDirection: "column",
  gap: "16px",
};

const freeCard: CSSProperties = {
  ...cardBase,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
};

// Fest dunkler Gradient statt theme.colors.chrome (kehrt sich zwischen
// Light/Dark-Mode um) - identisches Muster wie BillingPage.tsx::proCard,
// damit der feste weiße Text auf der Karte in beiden Modi lesbar bleibt.
const proCard: CSSProperties = {
  ...cardBase,
  background: "linear-gradient(135deg, #1c1c1f 0%, #2a2a2e 60%, #3a3a3f 100%)",
  border: `1px solid ${theme.colors.chromeBorder}`,
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.4)",
};

const planNameRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  gap: "10px",
};

const planName: CSSProperties = {
  fontSize: "1.3rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
};

const planTagMuted: CSSProperties = {
  padding: "5px 12px",
  borderRadius: theme.radius.pill,
  fontSize: "0.76rem",
  fontWeight: 700,
  color: theme.colors.textMuted,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.border}`,
};

const planTagBright: CSSProperties = {
  padding: "5px 12px",
  borderRadius: theme.radius.pill,
  fontSize: "0.76rem",
  fontWeight: 700,
  color: "#ffffff",
  background: "rgba(212, 212, 216, 0.16)",
  border: "1px solid rgba(212, 212, 216, 0.3)",
};

const planPrice: CSSProperties = {
  fontSize: "2.4rem",
  fontWeight: 900,
  color: theme.colors.textPrimary,
};

const planDescription: CSSProperties = {
  margin: 0,
  color: theme.colors.textSecondary,
  fontSize: "0.94rem",
  lineHeight: 1.6,
};

const planDescriptionBright: CSSProperties = {
  margin: 0,
  color: "rgba(255,255,255,0.78)",
  fontSize: "0.94rem",
  lineHeight: 1.6,
};

const featureList: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "10px",
};

const featureItem: CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  gap: "9px",
  fontSize: "0.9rem",
  color: theme.colors.textSecondary,
};

const featureItemBright: CSSProperties = {
  display: "flex",
  alignItems: "flex-start",
  gap: "9px",
  fontSize: "0.9rem",
  color: "rgba(255,255,255,0.92)",
};

const proPriceBlock: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "6px",
};

const proMainPrice: CSSProperties = {
  fontSize: "2.4rem",
  fontWeight: 900,
  color: "#ffffff",
};

const proYearlyPriceWrap: CSSProperties = {
  display: "flex",
  alignItems: "baseline",
  gap: "10px",
};

const proYearlyOldPrice: CSSProperties = {
  fontSize: "1.1rem",
  fontWeight: 700,
  color: "rgba(255,255,255,0.5)",
  textDecoration: "line-through",
};

const proSavingsRow: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "10px",
  flexWrap: "wrap",
};

const proSavingsBadge: CSSProperties = {
  padding: "3px 10px",
  borderRadius: theme.radius.pill,
  fontSize: "0.76rem",
  fontWeight: 800,
  background: "rgba(255,255,255,0.16)",
  color: "#ffffff",
};

const proSavingsText: CSSProperties = {
  fontSize: "0.84rem",
  color: "rgba(255,255,255,0.7)",
};

const proMonthlyHint: CSSProperties = {
  fontSize: "0.84rem",
  color: "rgba(255,255,255,0.7)",
};

const ctaBase: CSSProperties = {
  textDecoration: "none",
  textAlign: "center",
  padding: "13px 20px",
  borderRadius: theme.radius.pill,
  fontWeight: 700,
  fontSize: "0.95rem",
  marginTop: "6px",
};

const freeCta: CSSProperties = {
  ...ctaBase,
  color: theme.colors.textPrimary,
  background: theme.colors.panelAlt,
  border: `1px solid ${theme.colors.chromeBorder}`,
};

const proCta: CSSProperties = {
  ...ctaBase,
  color: "#ffffff",
  background: "rgba(255,255,255,0.14)",
  border: "1px solid rgba(255,255,255,0.3)",
};

const disclaimerText: CSSProperties = {
  marginTop: "36px",
  textAlign: "center",
  color: theme.colors.textMuted,
  fontSize: "0.84rem",
  lineHeight: 1.6,
};
