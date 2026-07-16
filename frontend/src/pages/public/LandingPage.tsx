import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  ArrowRight,
  CheckCircle2,
  Coins,
  Database,
  LayoutGrid,
  ShieldCheck,
  SlidersHorizontal,
  TrendingUp,
  Wrench,
} from "lucide-react";
import { theme } from "../../components/ui/theme";
import ParticleBeamBackground from "../../components/landing/ParticleBeamBackground";
import FloatingPanel from "../../components/landing/FloatingPanel";
import FloatingLabelCard from "../../components/landing/FloatingLabelCard";
import HeroIconRail from "../../components/landing/HeroIconRail";
import AudienceTabs, { type AudienceTab } from "../../components/landing/AudienceTabs";
import StatReadout from "../../components/landing/StatReadout";
import SectionHeading from "../../components/landing/SectionHeading";
import FeatureCard from "../../components/landing/FeatureCard";
import StepCard from "../../components/landing/StepCard";
import { useIsMobile, useIsTablet } from "../../hooks/useMediaQuery";

/** The app has no global sans-serif font-family (only "Instrument Serif" is
 * loaded), so headings fall back to the browser's default serif. That's a
 * pre-existing, site-wide condition out of scope to fix here — but the new
 * cinematic/architecture hero needs a clean grotesk look, so this hero's own
 * bold headings opt into a system sans stack explicitly. */
const heroSansFont =
  '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

function StaggerGroup({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      initial="hidden"
      whileInView="visible"
      viewport={{ once: true, margin: "-60px" }}
      transition={{ staggerChildren: theme.motion.stagger }}
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))",
        gap: "22px",
      }}
    >
      {children}
    </motion.div>
  );
}

/** Real analysis modes (mirrors `AnalyzePage.tsx`'s `modeOptions`) — the
 * hero's central panel and tab control are driven by this, not invented
 * marketing copy. */
const audienceTabs: AudienceTab[] = [
  {
    id: "full",
    label: "Vollanalyse",
    icon: <LayoutGrid size={16} />,
    benefit: "Ein vollständiges Bild des Unternehmens, ohne jede Methode einzeln aufzusetzen.",
    description:
      "Alle Analysemethoden in einem strukturierten Durchlauf kombiniert — ein vollständiges Bild statt Einzelauswertungen.",
  },
  {
    id: "wachstumswerte",
    label: "Wachstumswerte",
    icon: <TrendingUp size={16} />,
    benefit: "Erkennt Wachstumsqualität, bevor du jede Kennzahl einzeln nachrechnest.",
    description:
      "Fokussierte Auswertung von Wachstum, Qualität und Skalierbarkeit eines Unternehmens.",
  },
  {
    id: "dividendenwerte",
    label: "Dividendenwerte",
    icon: <Coins size={16} />,
    benefit: "Prüft Ausschüttungsstabilität automatisiert, statt jede Bilanz selbst zu durchsuchen.",
    description:
      "Strukturierte Prüfung von Ausschüttungsstabilität, Substanz und Dividendenhistorie.",
  },
  {
    id: "custom",
    label: "Eigene Analyse",
    icon: <SlidersHorizontal size={16} />,
    benefit: "Baue einmal deine eigene Logik und nutze sie für jede künftige Analyse wieder.",
    description:
      "Eigene Kennzahlen-Kombinationen frei zusammenstellen und für jede künftige Analyse wiederverwenden.",
  },
];

const workflowSteps = [
  "Unternehmen auswählen",
  "Analysemethode starten",
  "Kennzahlen automatisiert interpretieren",
  "Ergebnisse strukturiert vergleichen",
];

const AUTO_ROTATE_MS = 5000;

function LandingPage() {
  const [activeTabId, setActiveTabId] = useState(audienceTabs[0].id);
  const activeTab = audienceTabs.find((tab) => tab.id === activeTabId) ?? audienceTabs[0];
  const isTablet = useIsTablet();
  const isMobile = useIsMobile();
  const isPausedRef = useRef(false);

  useEffect(() => {
    const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)")?.matches ?? false;
    if (prefersReducedMotion) return;

    const interval = window.setInterval(() => {
      if (isPausedRef.current) return;
      setActiveTabId((current) => {
        const index = audienceTabs.findIndex((tab) => tab.id === current);
        const next = audienceTabs[(index + 1) % audienceTabs.length];
        return next.id;
      });
    }, AUTO_ROTATE_MS);

    return () => window.clearInterval(interval);
  }, [activeTabId]);

  return (
    <div style={{ width: "100%", position: "relative" }}>
      {/* Spans the full page (not just the hero) so the particle field
       * keeps flowing behind every section down to the footer. */}
      <ParticleBeamBackground />

      {/* Hero Section — cinematic, full-bleed */}
      <section
        style={
          isMobile
            ? { ...heroSection, padding: "88px 20px 32px" }
            : heroSection
        }
      >
        <div aria-hidden="true" style={gridOverlay} />

        <HeroIconRail items={audienceTabs} activeId={activeTabId} onChange={setActiveTabId} />

        <div style={isMobile ? { ...heroContent, gap: "32px" } : heroContent}>
          <motion.div
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
            style={{ textAlign: "center" }}
          >
            <div style={isMobile ? { ...heroEyebrow, marginBottom: "16px" } : heroEyebrow}>
              Fundamentalanalyse · neu gedacht
            </div>

            <h1 style={isMobile ? { ...heroHeadline, marginBottom: "14px" } : heroHeadline}>
              ALLE FUNDAMENTALDATEN.
              <br />
              EINE QUELLE.
              <br />
              <span
                style={{
                  background: theme.gradients.ctaPrimary,
                  WebkitBackgroundClip: "text",
                  backgroundClip: "text",
                  color: "transparent",
                }}
              >
                JEDE ZAHL NACHVOLLZIEHBAR.
              </span>
            </h1>

            <p
              style={
                isMobile
                  ? { ...heroSubtext, marginBottom: "20px", lineHeight: 1.6 }
                  : heroSubtext
              }
            >
              ComAnalysis bündelt SEC-Originaldaten und transparent berechnete
              Kennzahlen in einem Werkzeug. Zu jeder Zahl siehst du, woher sie
              stammt und wie sie berechnet wurde. Die Interpretation — und
              jede Entscheidung — bleibt bei dir.
            </p>

            <div style={isMobile ? { ...heroCtaRow, marginBottom: "16px" } : heroCtaRow}>
              <motion.div
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
                transition={theme.motion.spring}
                style={{ display: "inline-block" }}
              >
                <Link to="/register?src=hero" style={primaryCta}>
                  Kostenlos starten — 50 Analyse-Einheiten/Monat inklusive
                  <ArrowRight size={18} />
                </Link>
              </motion.div>
              <motion.div
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
                transition={theme.motion.spring}
                style={{ display: "inline-block" }}
              >
                <Link to="/login" style={secondaryCta}>
                  Bereits Zugang? Login
                </Link>
              </motion.div>
            </div>

            <Link
              to="/pricing"
              style={isMobile ? { ...pricingInlineLink, marginBottom: "14px" } : pricingInlineLink}
            >
              Was kostet Pro? Preise ansehen →
            </Link>

            <div style={checklistRow}>
              <span style={checklistItem}>
                <CheckCircle2 size={16} color={theme.colors.success} style={checklistIcon} /> Fundamentaldaten direkt aus SEC-Filings, Kursdaten von etablierten Marktdatenanbietern
              </span>
              <span style={checklistItem}>
                <CheckCircle2 size={16} color={theme.colors.success} style={checklistIcon} /> Jede Formel offen einsehbar
              </span>
              <span style={checklistItem}>
                <CheckCircle2 size={16} color={theme.colors.success} style={checklistIcon} /> Keine Anlageberatung. Nur transparent berechnete Kennzahlen.
              </span>
            </div>
          </motion.div>

          <div style={panelStack}>
            {/* Central mode panel — solo row, no longer stretched to match
             * the side cards' height in a shared grid track. */}
            <FloatingPanel seed={0} style={centralPanelFloatWrap}>
              <div
                style={centralPanel}
                onMouseEnter={() => {
                  isPausedRef.current = true;
                }}
                onMouseLeave={() => {
                  isPausedRef.current = false;
                }}
              >
                <div style={centralPanelEyebrow}>{activeTab.label}</div>
                <p style={centralPanelBenefit}>{activeTab.benefit}</p>
                <p style={centralPanelDescription}>{activeTab.description}</p>

                <div style={workflowList}>
                  {workflowSteps.map((step, index) => (
                    <div key={step} style={workflowRow}>
                      <span style={workflowIndex}>{index + 1}</span>
                      <span style={workflowText}>{step}</span>
                    </div>
                  ))}
                </div>

                <div style={{ marginTop: "20px" }}>
                  <AudienceTabs tabs={audienceTabs} activeId={activeTabId} onChange={setActiveTabId} />
                </div>
              </div>
            </FloatingPanel>

            {/* Both side cards share one FloatingPanel so they float in
             * sync ("together"), tilted outward into a fanned pair. */}
            <FloatingPanel seed={1} style={isTablet || isMobile ? duoRowStacked : duoRowFanned}>
              <div style={isTablet || isMobile ? undefined : tiltLeft}>
                <FloatingLabelCard
                  icon={<Database size={16} />}
                  title="Kennzahlen-Engine"
                  description="Automatisierte Auswertung von Fundamentaldaten statt manueller Tabellenarbeit."
                  secondaryDescription="Parst SEC-Filings und Marktdaten automatisch in strukturierte Kennzahlen."
                  style={isTablet || isMobile ? { ...duoCard, maxWidth: "100%" } : duoCard}
                />
              </div>
              <div style={isTablet || isMobile ? undefined : tiltRight}>
                <FloatingLabelCard
                  icon={<Wrench size={16} />}
                  title="Eigene Logik"
                  description="Baue individuelle Analysen aus deinen eigenen Kennzahlen-Kombinationen."
                  secondaryDescription="Eigene Kennzahlen-Definitionen, einmal gebaut und für jede Analyse wiederverwendbar."
                  style={isTablet || isMobile ? { ...duoCard, maxWidth: "100%" } : duoCard}
                />
              </div>
            </FloatingPanel>
          </div>

          <div style={footerStatsRow}>
            <div style={dataSourcesStrip}>
              <span style={dataSourcesLabel}>Datenbasis</span>
              {["SEC Filings", "Marktdaten", "Eigene Kennzahlen-Engine"].map((source) => (
                <span key={source} style={dataSourcePill}>
                  {source}
                </span>
              ))}
            </div>

            <div style={statReadoutRow}>
              <StatReadout label="Methoden" value="8 vordefiniert" />
              <StatReadout label="Builder" value="Eigene Logik inklusive" />
              <StatReadout label="Vergleich" value="Mehrere Unternehmen parallel" />
            </div>
          </div>
        </div>
      </section>

      {/* Value Section */}
      <section style={sectionWrapper}>
        <SectionHeading
          eyebrow="Warum ComAnalysis"
          title="Eine Plattform statt drei Werkzeuge"
          subtitle="Vordefinierte Methoden, ein eigener Logik-Builder und echte Produktinfrastruktur — kombiniert, damit du schneller zu belastbaren Entscheidungsgrundlagen kommst."
        />

        <StaggerGroup>
          <FeatureCard
            icon={<LayoutGrid size={20} />}
            title="Vordefinierte Methoden"
            text="Acht etablierte Analysemodi — von Wachstumswerten bis Turnarounds — ersetzen den manuellen Aufbau jeder einzelnen Auswertung."
          />
          <FeatureCard
            icon={<SlidersHorizontal size={20} />}
            title="Eigene Logik bauen"
            text="Kombiniere eigene Kennzahlen zu individuellen Analysebausteinen und nutze sie wiederholt für jedes Unternehmen."
          />
          <FeatureCard
            icon={<ShieldCheck size={20} />}
            title="Vergleich statt Tabellen-Jonglage"
            text="Mehrere Unternehmen nebeneinander vergleichen und Kennzahlen über historische Zeiträume einordnen — ohne zwischen Tools zu wechseln."
          />
        </StaggerGroup>
      </section>

      {/* How it works */}
      <section style={sectionWrapper}>
        <div style={howItWorksPanel}>
          <SectionHeading
            align="left"
            eyebrow="So funktioniert die App"
            title="Vom Login zur strukturierten Analyse"
            subtitle="Nutzer werden von der LandingPage über Login und Registrierung in das Analyse-Dashboard geführt. Dort werden Analysen gestartet, eigene Methoden erstellt und das Abo verwaltet."
          />

          <div style={stepRowWrapper}>
            <div aria-hidden="true" style={stepConnectorLine} />
            <StaggerGroup>
              {[
                "Unternehmen oder Symbol suchen",
                "Analysemethode wählen oder eigene Logik bauen",
                "Ergebnis mit Quelle und Rechenweg prüfen",
                "Mit anderen Unternehmen vergleichen oder speichern",
              ].map((step, index) => (
                <StepCard key={step} index={index} text={step} />
              ))}
            </StaggerGroup>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section style={sectionWrapper}>
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.55, ease: [0.4, 0, 0.2, 1] }}
          style={finalCtaPanel}
        >
          <div aria-hidden="true" style={finalCtaGlow} />

          <h2 style={finalCtaTitle}>Bereit für deine erste strukturierte Analyse?</h2>

          <p style={finalCtaText}>
            Registriere dich und starte deine erste Analyse mit 50
            Analyse-Einheiten kostenlos — jede Kennzahl mit Quelle und
            Rechenweg.
          </p>

          <div style={isMobile ? finalCtaButtonRowMobile : finalCtaButtonRow}>
            <motion.div
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              transition={theme.motion.spring}
              style={isMobile ? { display: "block", width: "100%" } : { display: "inline-block" }}
            >
              <Link
                to="/register?src=final"
                style={isMobile ? { ...finalCtaPrimary, display: "block", width: "100%", boxSizing: "border-box" } : finalCtaPrimary}
              >
                Kostenlos registrieren
              </Link>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
              transition={theme.motion.spring}
              style={
                isMobile
                  ? { display: "block", width: "100%", marginTop: "14px" }
                  : { display: "inline-block" }
              }
            >
              <Link
                to="/login"
                style={isMobile ? { ...finalCtaSecondary, display: "block", width: "100%", boxSizing: "border-box" } : finalCtaSecondary}
              >
                Bereits registriert? Login
              </Link>
            </motion.div>
          </div>
        </motion.div>
      </section>
    </div>
  );
}

/* styles */

const heroSection: React.CSSProperties = {
  position: "relative",
  overflow: "hidden",
  // svh statt vh: rechnet gegen den KLEINEN iOS-Safari-Viewport (Toolbar
  // eingeblendet) statt den großen, damit der Hero-CTA auch mit sichtbarer
  // Toolbar in den ersten Viewport passt (RESPONSIVE.md R-P1-7).
  minHeight: "94svh",
  display: "flex",
  alignItems: "center",
  padding: "120px 24px 48px",
};

const gridOverlay: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  zIndex: 1,
  pointerEvents: "none",
  backgroundImage:
    "repeating-linear-gradient(0deg, rgba(255,255,255,0.05) 0px, rgba(255,255,255,0.05) 1px, transparent 1px, transparent 64px), repeating-linear-gradient(90deg, rgba(255,255,255,0.05) 0px, rgba(255,255,255,0.05) 1px, transparent 1px, transparent 64px)",
  maskImage: "radial-gradient(ellipse 70% 60% at 50% 40%, rgba(0,0,0,0.9), transparent 75%)",
  WebkitMaskImage: "radial-gradient(ellipse 70% 60% at 50% 40%, rgba(0,0,0,0.9), transparent 75%)",
};

const heroContent: React.CSSProperties = {
  position: "relative",
  zIndex: 2,
  width: "100%",
  maxWidth: "1180px",
  margin: "0 auto",
  display: "flex",
  flexDirection: "column",
  gap: "56px",
};

const heroEyebrow: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  padding: "9px 18px",
  borderRadius: theme.radius.pill,
  background: theme.colors.chromeSoft,
  border: `1px solid ${theme.colors.chromeBorder}`,
  color: theme.colors.chrome,
  fontWeight: 700,
  fontSize: "0.82rem",
  fontFamily: heroSansFont,
  letterSpacing: "0.06em",
  textTransform: "uppercase",
  marginBottom: "28px",
};

const heroHeadline: React.CSSProperties = {
  margin: "0 0 24px 0",
  // Min-Bound abgesenkt (war 2.4rem): "FUNDAMENTALDATEN." ist ein langes
  // Einzelwort, das bei der alten Untergrenze auf schmalen Viewports
  // (<420px) horizontal überlief, da 5.4vw dort schon unter 2.4rem liegt
  // und der Clamp auf die Untergrenze zurückfällt.
  fontSize: "clamp(1.7rem, 5.4vw, 4.4rem)",
  lineHeight: 1.08,
  letterSpacing: "-0.03em",
  fontWeight: 800,
  fontFamily: heroSansFont,
  color: theme.colors.textPrimary,
  textTransform: "uppercase",
};

const heroSubtext: React.CSSProperties = {
  margin: "0 auto 34px",
  fontSize: "1.14rem",
  lineHeight: 1.85,
  color: theme.colors.textSecondary,
  maxWidth: "680px",
};

const heroCtaRow: React.CSSProperties = {
  display: "flex",
  gap: "16px",
  flexWrap: "wrap",
  justifyContent: "center",
  marginBottom: "30px",
};

const primaryCta: React.CSSProperties = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  gap: "8px",
  background: theme.gradients.ctaPrimary,
  color: theme.colors.bgDeep,
  fontWeight: 700,
  fontSize: "1.02rem",
  padding: "15px 24px",
  borderRadius: theme.radius.pill,
  boxShadow: "0 16px 34px rgba(0, 0, 0, 0.45)",
};

const secondaryCta: React.CSSProperties = {
  textDecoration: "none",
  display: "inline-flex",
  alignItems: "center",
  background: theme.glass.subtle.background,
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "1.02rem",
  padding: "15px 24px",
  borderRadius: theme.radius.pill,
  border: `1px solid ${theme.glass.subtle.border}`,
};

const pricingInlineLink: React.CSSProperties = {
  display: "block",
  textAlign: "center",
  color: theme.colors.textSecondary,
  fontSize: "0.9rem",
  fontWeight: 600,
  textDecoration: "none",
  marginBottom: "24px",
};

const checklistRow: React.CSSProperties = {
  display: "flex",
  gap: "24px",
  flexWrap: "wrap",
  justifyContent: "center",
  color: theme.colors.textMuted,
  fontSize: "0.98rem",
  lineHeight: 1.75,
};

const checklistItem: React.CSSProperties = {
  display: "inline-flex",
  // flex-start statt center: bei zweizeilig umgebrochenem Text (schmale
  // Mobile-/Tablet-Breiten) zentrierte sich das Icon sonst gegen die GANZE
  // Textblock-Höhe und "schwebte" sichtbar losgelöst von der ersten Zeile
  // statt sauber neben ihr zu sitzen.
  alignItems: "flex-start",
  gap: "8px",
  textAlign: "left",
};

// Optischer Ausgleich zur Cap-Height der ersten Textzeile + garantiert
// unveränderte Icon-Größe im Flex-Layout.
const checklistIcon: React.CSSProperties = {
  flexShrink: 0,
  marginTop: "3px",
};

const panelStack: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  gap: "28px",
};

const centralPanelFloatWrap: React.CSSProperties = {
  width: "100%",
  maxWidth: "760px",
  position: "relative",
  zIndex: 2,
};

const duoRowFanned: React.CSSProperties = {
  display: "flex",
  justifyContent: "center",
  alignItems: "flex-start",
  gap: "24px",
  width: "100%",
  maxWidth: "920px",
  marginTop: "-16px",
  position: "relative",
  zIndex: 1,
};

const duoRowStacked: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  gap: "20px",
  width: "100%",
};

const duoCard: React.CSSProperties = {
  flex: "1 1 280px",
  maxWidth: "340px",
};

const tiltLeft: React.CSSProperties = {
  transform: "rotate(-4deg) translateY(6px)",
};

const tiltRight: React.CSSProperties = {
  transform: "rotate(4deg) translateY(6px)",
};

const centralPanel: React.CSSProperties = {
  background: theme.glass.elevated.background,
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  borderRadius: theme.radius.lg,
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
  padding: "28px",
};

const centralPanelEyebrow: React.CSSProperties = {
  fontSize: "1.1rem",
  fontWeight: 800,
  fontFamily: heroSansFont,
  color: theme.colors.textPrimary,
  marginBottom: "10px",
};

const centralPanelBenefit: React.CSSProperties = {
  margin: "0 0 12px 0",
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "1.02rem",
  lineHeight: 1.5,
};

const centralPanelDescription: React.CSSProperties = {
  margin: "0 0 18px 0",
  color: theme.colors.textSecondary,
  fontSize: "0.96rem",
  lineHeight: 1.7,
};

const workflowList: React.CSSProperties = {
  display: "grid",
  gap: "10px",
};

const workflowRow: React.CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: "12px",
  background: "rgba(255,255,255,0.04)",
  border: `1px solid ${theme.glass.subtle.border}`,
  borderRadius: theme.radius.md,
  padding: "11px 14px",
};

const workflowIndex: React.CSSProperties = {
  width: "26px",
  height: "26px",
  borderRadius: theme.radius.pill,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  background: theme.colors.chromeSoft,
  color: theme.colors.chrome,
  fontWeight: 800,
  fontSize: "0.82rem",
  flexShrink: 0,
};

const workflowText: React.CSSProperties = {
  color: theme.colors.textSecondary,
  fontWeight: 600,
  fontSize: "0.92rem",
  // Flex items default to min-width:auto, which stops text from wrapping
  // below its unwrapped width — minWidth:0 lets it wrap normally and stops
  // this row (and the whole central panel) from forcing horizontal overflow
  // on narrow viewports.
  minWidth: 0,
};

const footerStatsRow: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  justifyContent: "space-between",
  alignItems: "center",
  gap: "20px",
};

const dataSourcesStrip: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  alignItems: "center",
  gap: "8px",
};

const dataSourcesLabel: React.CSSProperties = {
  fontSize: "0.74rem",
  fontWeight: 700,
  letterSpacing: "0.05em",
  textTransform: "uppercase",
  color: theme.colors.textMuted,
  marginRight: "4px",
};

const dataSourcePill: React.CSSProperties = {
  padding: "6px 12px",
  borderRadius: theme.radius.pill,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textSecondary,
  fontSize: "0.78rem",
  fontWeight: 600,
};

const statReadoutRow: React.CSSProperties = {
  display: "flex",
  flexWrap: "wrap",
  gap: "24px",
};

const sectionWrapper: React.CSSProperties = {
  maxWidth: "1280px",
  margin: "0 auto",
  padding: "48px 24px",
};

const howItWorksPanel: React.CSSProperties = {
  background: theme.glass.elevated.background,
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  borderRadius: theme.radius.lg,
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
  padding: "40px",
};

const stepRowWrapper: React.CSSProperties = {
  position: "relative",
};

const stepConnectorLine: React.CSSProperties = {
  position: "absolute",
  top: "18px",
  left: "5%",
  right: "5%",
  height: "1px",
  background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.18), transparent)",
  zIndex: 0,
};

const finalCtaPanel: React.CSSProperties = {
  position: "relative",
  overflow: "hidden",
  background: theme.gradients.ctaPrimary,
  borderRadius: theme.radius.lg,
  padding: "48px 36px",
  textAlign: "center",
  color: theme.colors.bgDeep,
  boxShadow: "0 28px 70px rgba(0, 0, 0, 0.5)",
};

const finalCtaGlow: React.CSSProperties = {
  position: "absolute",
  top: "-40%",
  left: "50%",
  transform: "translateX(-50%)",
  width: "70%",
  height: "140%",
  background: "radial-gradient(circle, rgba(255,255,255,0.5), transparent 65%)",
  pointerEvents: "none",
};

const finalCtaTitle: React.CSSProperties = {
  position: "relative",
  margin: "0 0 18px 0",
  fontSize: "2.4rem",
  letterSpacing: "-0.04em",
  lineHeight: 1.25,
  fontFamily: heroSansFont,
};

const finalCtaText: React.CSSProperties = {
  position: "relative",
  margin: "0 auto 28px",
  maxWidth: "820px",
  lineHeight: 1.9,
  fontSize: "1.1rem",
};

const finalCtaButtonRow: React.CSSProperties = {
  position: "relative",
  display: "flex",
  justifyContent: "center",
  gap: "14px",
  flexWrap: "wrap",
};

// Eigenständiges Mobile-Layout statt sich auf Flex-Wrap-Zeilenumbruch mit
// row-gap zu verlassen: beide CTAs wurden auf schmalen Screens live
// überlappend beobachtet. Echtes Block-Stacking mit explizitem marginTop
// (statt gap) schließt jede Überlappung unabhängig von Zeilenumbruch-
// Timing aus — volle Breite macht die Buttons nebenbei präsenter/stärker.
const finalCtaButtonRowMobile: React.CSSProperties = {
  position: "relative",
  display: "block",
  width: "100%",
};

const finalCtaPrimary: React.CSSProperties = {
  textDecoration: "none",
  background: theme.colors.bgDeep,
  color: theme.colors.textPrimary,
  fontWeight: 800,
  fontSize: "1.02rem",
  padding: "15px 24px",
  borderRadius: theme.radius.pill,
};

const finalCtaSecondary: React.CSSProperties = {
  textDecoration: "none",
  background: "transparent",
  color: theme.colors.bgDeep,
  fontWeight: 700,
  fontSize: "1.02rem",
  padding: "15px 24px",
  borderRadius: theme.radius.pill,
  border: "1px solid rgba(0,0,0,0.35)",
};

export default LandingPage;
