import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowUpRight, BarChart3, History, RotateCcw, Sparkles } from "lucide-react";
import { getCurrentUser } from "../../api/auth";
import {
  getAnalysisHistory,
  getAnalysisHistorySnapshot,
  type AnalysisHistoryEntry,
  type AnalysisHistorySnapshot,
} from "../../api/analysis";
import { getCustomMetricsCatalog } from "../../api/customAnalysis";
import type { MetricCatalogEntry, MetricSelection, CustomMetricResult } from "../../types/customAnalysis";
import type { CategoryResult } from "../../types/analysis";
import { theme } from "../../components/ui/theme";
import ParallaxCard from "../../components/ui/ParallaxCard";
import AnimatedNumber from "../../components/ui/AnimatedNumber";
import Modal from "../../components/ui/Modal";
import StackedCards from "../../components/ui/StackedCards";
import ParticleBeamBackground from "../../components/landing/ParticleBeamBackground";
import LivePriceBadge from "../../components/shared/LivePriceBadge";
import DataSourceStatusWidget from "../../components/shared/DataSourceStatusWidget";
import AnalyzeResultsDashboard from "../../components/analysis/AnalyzeResultsDashboard";
import CustomAnalysisResultsList from "../../components/customAnalysis/CustomAnalysisResultsList";

const HISTORY_PREVIEW_COUNT = 5;

const HISTORY_MODE_LABELS: Record<string, string> = {
  full: "Vollanalyse",
  wachstumswerte: "Wachstumswerte",
  dividendenwerte: "Dividendenwerte",
  "average-grower": "Average Grower",
  "typische-zykliker": "Typische Zykliker",
  turnarounds: "Turnarounds",
  optionality: "Optionalitäten",
  "asset-play": "Asset Play",
};

const HISTORY_STATUS_LABELS: Record<string, string> = {
  running: "Läuft...",
  done: "Abgeschlossen",
  error: "Fehler",
};

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0 },
};

function DashboardPage() {
  const navigate = useNavigate();
  const [greetingName, setGreetingName] = useState<string | null>(null);
  const [currentPlan, setCurrentPlan] = useState("free");
  const [billingStatus, setBillingStatus] = useState("inactive");
  const [monthlyCount, setMonthlyCount] = useState(0);
  const [monthlyLimit, setMonthlyLimit] = useState<number | null>(null);
  const [isLoadingUser, setIsLoadingUser] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  const [recentAnalyses, setRecentAnalyses] = useState<AnalysisHistoryEntry[]>([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [isHistoryModalOpen, setIsHistoryModalOpen] = useState(false);

  const [snapshotEntry, setSnapshotEntry] = useState<AnalysisHistoryEntry | null>(null);
  const [snapshotData, setSnapshotData] = useState<AnalysisHistorySnapshot | null>(null);
  const [isLoadingSnapshot, setIsLoadingSnapshot] = useState(false);
  const [snapshotErrorMessage, setSnapshotErrorMessage] = useState("");
  const [snapshotCatalog, setSnapshotCatalog] = useState<MetricCatalogEntry[]>([]);
  const [rerunErrorMessage, setRerunErrorMessage] = useState("");

  const sortedAnalyses = useMemo(
    () =>
      [...recentAnalyses].sort(
        (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      ),
    [recentAnalyses]
  );
  const previewAnalyses = sortedAnalyses.slice(0, HISTORY_PREVIEW_COUNT);

  const normalizedPlan = currentPlan.trim().toLowerCase();
  const normalizedBillingStatus = billingStatus.trim().toLowerCase();

  const isFreePlan = normalizedPlan === "free";
  const displayPlan = isFreePlan ? "Free" : "Pro";

  const displayBillingStatus =
    normalizedBillingStatus === "active" ? "Aktiv" : "Inaktiv";

  const billingStatusColor =
    normalizedBillingStatus === "active" ? theme.colors.successText : theme.colors.dangerText;

  useEffect(() => {
    async function loadCurrentUser() {
      try {
        setErrorMessage("");

        const user = await getCurrentUser();

        setGreetingName(user.username ?? user.first_name ?? null);

        setCurrentPlan(typeof user.plan === "string" ? user.plan : "free");

        setBillingStatus(
          typeof user.billing_status === "string" ? user.billing_status : "inactive"
        );

        setMonthlyCount(
          typeof user.monthly_request_count === "number" ? user.monthly_request_count : 0
        );

        setMonthlyLimit(
          typeof user.monthly_request_limit === "number" ? user.monthly_request_limit : null
        );
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

  useEffect(() => {
    async function loadHistory() {
      try {
        const history = await getAnalysisHistory();
        setRecentAnalyses(history);
      } catch {
        setRecentAnalyses([]);
      } finally {
        setIsLoadingHistory(false);
      }
    }

    loadHistory();
  }, []);

  async function handleViewSnapshot(entry: AnalysisHistoryEntry) {
    setSnapshotEntry(entry);
    setSnapshotData(null);
    setSnapshotErrorMessage("");
    setIsLoadingSnapshot(true);

    try {
      const data = await getAnalysisHistorySnapshot(entry.id);
      setSnapshotData(data);

      if (entry.mode === "custom" && snapshotCatalog.length === 0) {
        try {
          setSnapshotCatalog(await getCustomMetricsCatalog());
        } catch {
          setSnapshotCatalog([]);
        }
      }
    } catch {
      setSnapshotErrorMessage("Snapshot konnte nicht geladen werden.");
    } finally {
      setIsLoadingSnapshot(false);
    }
  }

  async function handleRerun(entry: AnalysisHistoryEntry) {
    setRerunErrorMessage("");

    if (entry.mode === "custom" && entry.definition_id == null) {
      try {
        const data = await getAnalysisHistorySnapshot(entry.id);
        const selection = (data.result_snapshot as { selection?: MetricSelection[] } | null)
          ?.selection;

        if (!selection || selection.length === 0) {
          setRerunErrorMessage(
            "Für diese einmalige Analyse sind keine Metriken mehr gespeichert."
          );
          return;
        }

        navigate("/app/analyze", {
          state: { rerun: { mode: entry.mode, symbol: entry.symbol, metrics: selection } },
        });
      } catch {
        setRerunErrorMessage("Analyse konnte nicht erneut gestartet werden.");
      }
      return;
    }

    navigate("/app/analyze", {
      state: {
        rerun: {
          mode: entry.mode,
          symbol: entry.symbol,
          frequency: entry.frequency,
          definitionId: entry.definition_id,
        },
      },
    });
  }

  return (
    <div style={{ position: "relative", display: "flex", flexDirection: "column", gap: "40px", padding: "10px 6px" }}>
      <ParticleBeamBackground densityMultiplier={1.3} />

      {errorMessage ? (
        <div
          style={{
            padding: "14px 16px",
            borderRadius: theme.radius.md,
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

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "8px",
            marginBottom: "14px",
            padding: "8px 14px",
            borderRadius: theme.radius.pill,
            background: theme.colors.chromeSoft,
            border: `1px solid ${theme.colors.chromeBorder}`,
            color: theme.colors.chrome,
            fontSize: "0.86rem",
            fontWeight: 700,
            letterSpacing: "0.03em",
          }}
        >
          <Sparkles size={14} />
          Dein Analyse-Zentrum
        </div>

        <h1
          style={{
            margin: "0 0 14px 0",
            fontSize: "3rem",
            letterSpacing: "-0.045em",
            lineHeight: 1.05,
            color: theme.colors.textPrimary,
          }}
        >
          {greetingName ? `Willkommen zurück, ${greetingName}` : "Dashboard"}
        </h1>

        <p
          style={{
            margin: 0,
            color: theme.colors.textSecondary,
            fontSize: "1.14rem",
            lineHeight: 1.85,
            maxWidth: "860px",
          }}
        >
          Behalte deine Analysen, dein Konto und deine eigenen
          Bewertungsmethoden an einem Ort im Blick. Von hier aus startest du
          neue Analysen, entwickelst individuelle Analyse-Logiken und steuerst
          deinen gesamten Workflow deutlich strukturierter.
        </p>
      </motion.div>

      {/* PRIMARY GRID */}
      <motion.div
        initial="hidden"
        animate="visible"
        transition={{ staggerChildren: theme.motion.stagger, delayChildren: 0.1 }}
        style={{
          position: "relative",
          zIndex: 1,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "28px",
          alignItems: "start",
          marginBottom: "16px",
        }}
      >
        {/* MAIN ACTION */}
        <motion.div variants={fadeUp} transition={{ duration: 0.4 }}>
          <Link to="/app/analyze" style={cardLink} data-tour="dashboard-cta">
            <ParallaxCard
              tilt={7}
              lift={8}
              style={{
                background: theme.colors.panel,
                minHeight: "220px",
                justifyContent: "center",
                cursor: "pointer",
              }}
            >
              <div style={mainCardLabel}>Schnellstart</div>
              <div style={{ ...mainCardTitle, display: "flex", alignItems: "center", gap: "10px" }}>
                Analyse starten
                <ArrowUpRight size={22} color={theme.colors.chrome} />
              </div>
              <div style={mainCardText}>
                Wähle ein Unternehmen und starte direkt eine strukturierte
                Fundamentalanalyse mit deinen bevorzugten Methoden.
              </div>
            </ParallaxCard>
          </Link>
        </motion.div>

        {/* SIDE */}
        <div style={{ display: "flex", flexDirection: "column", gap: "22px" }}>
          <motion.div variants={fadeUp} transition={{ duration: 0.4 }}>
            <Link to="/app/compare" style={cardLink}>
              <ParallaxCard
                tilt={7}
                lift={8}
                style={{
                  background: theme.colors.panel,
                  minHeight: "150px",
                  justifyContent: "center",
                  cursor: "pointer",
                }}
              >
                <div style={mainCardLabel}>Vergleich</div>
                <div style={{ ...cardTitle, fontSize: "1.6rem", display: "flex", alignItems: "center", gap: "10px" }}>
                  Charts und Daten vergleichen
                  <BarChart3 size={20} color={theme.colors.chrome} />
                </div>
                <div style={cardText}>
                  Stelle Kennzahlen mehrerer Aktien gegenüber und vergleiche
                  Verläufe direkt im Chart.
                </div>
              </ParallaxCard>
            </Link>
          </motion.div>

          {!isLoadingUser && isFreePlan ? (
            <motion.div variants={fadeUp} transition={{ duration: 0.4 }}>
              <Link to="/app/billing" style={cardLink}>
                <motion.div
                  whileHover={{ y: -4 }}
                  transition={theme.motion.spring}
                  style={upgradeCard}
                >
                  <div style={cardLabelLight}>Pro</div>
                  <div style={cardTitleBright}>Upgrade to Pro</div>
                  <div style={cardTextBright}>
                    Mehr Analysen & erweiterte Features freischalten.
                  </div>
                </motion.div>
              </Link>
            </motion.div>
          ) : isLoadingUser ? (
            <div style={loadingCard}>
              <div style={cardLabel}>Lädt</div>
              <div style={cardTitle}>Tarif wird geladen</div>
              <div style={cardText}>
                Deine Account- und Tarifinformationen werden abgerufen.
              </div>
            </div>
          ) : null}
        </div>
      </motion.div>

      {/* LOWER GRID */}
      <motion.div
        initial="hidden"
        animate="visible"
        transition={{ staggerChildren: theme.motion.stagger, delayChildren: 0.25 }}
        style={{
          position: "relative",
          zIndex: 0,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: "22px",
          alignItems: "stretch",
        }}
      >
        {/* ANALYSIS HISTORY */}
        <motion.div variants={fadeUp} transition={{ duration: 0.4 }} style={panel} data-tour="dashboard-analysis-history">
          <div style={panelTitle}>Letzte Analysen</div>

          {isLoadingHistory ? (
            <div style={emptyState}>Wird geladen...</div>
          ) : sortedAnalyses.length === 0 ? (
            <div style={emptyState}>
              Noch keine Analysen vorhanden. Starte deine erste Analyse und sie
              erscheint hier.
            </div>
          ) : (
            <>
              <StackedCards
                items={previewAnalyses}
                getKey={(entry) => entry.job_id}
                maxPeek={3}
                expandLabel={(hidden) => `${hidden} weitere Analysen anzeigen`}
                renderCard={(entry) => (
                  <div style={historyStackCard}>
                    <div style={{ ...accountRow, marginBottom: 0, flexWrap: "wrap", gap: "8px 14px" }}>
                      <span
                        style={{
                          ...accountLabel,
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "10px",
                          minWidth: 0,
                          overflow: "hidden",
                          whiteSpace: "nowrap",
                          textOverflow: "ellipsis",
                        }}
                      >
                        <span style={{ overflow: "hidden", textOverflow: "ellipsis" }}>
                          {entry.symbol} · {HISTORY_MODE_LABELS[entry.mode] ?? entry.mode}
                        </span>
                        <LivePriceBadge symbol={entry.symbol} size="sm" />
                      </span>
                      <span style={{ ...accountValue, flexShrink: 0 }}>
                        {HISTORY_STATUS_LABELS[entry.status] ?? entry.status}
                      </span>
                    </div>
                  </div>
                )}
              />

              {sortedAnalyses.length > HISTORY_PREVIEW_COUNT ? (
                <button
                  type="button"
                  onClick={() => setIsHistoryModalOpen(true)}
                  style={historyButton}
                >
                  <History size={15} />
                  Vollständige Historie ansehen
                </button>
              ) : null}
            </>
          )}
        </motion.div>

        {/* ACCOUNT */}
        <motion.div variants={fadeUp} transition={{ duration: 0.4 }} style={panel}>
          <div style={panelTitle}>Account</div>

          <div style={accountRow}>
            <span style={accountLabel}>Plan</span>
            <strong style={accountValue}>{isLoadingUser ? "Lädt..." : displayPlan}</strong>
          </div>

          <div style={accountRow}>
            <span style={accountLabel}>Status</span>
            <strong style={{ ...accountValue, color: isLoadingUser ? theme.colors.textPrimary : billingStatusColor }}>
              {isLoadingUser ? "Lädt..." : displayBillingStatus}
            </strong>
          </div>

          {!isLoadingUser && isFreePlan ? (
            <div style={accountRow}>
              <span style={accountLabel}>Limit</span>
              <strong style={accountValue}>
                {isLoadingUser ? (
                  "Lädt..."
                ) : (
                  <>
                    <AnimatedNumber value={monthlyCount} />/{monthlyLimit ?? "∞"}
                  </>
                )}
              </strong>
            </div>
          ) : null}

          {!isLoadingUser && isFreePlan && monthlyLimit && monthlyCount / monthlyLimit >= 0.8 ? (
            <div
              style={{
                marginTop: "10px",
                padding: "10px 12px",
                borderRadius: theme.radius.md,
                background: theme.colors.chromeSoft,
                border: `1px solid ${theme.colors.chromeBorder}`,
                color: theme.colors.textSecondary,
                fontSize: "0.85rem",
                lineHeight: 1.5,
              }}
            >
              Du hast {monthlyCount} von {monthlyLimit} Analysen diesen Monat
              genutzt.{" "}
              <Link to="/app/billing" style={{ color: theme.colors.chrome, fontWeight: 700 }}>
                Zu Pro wechseln
              </Link>{" "}
              für unbegrenzte Analysen.
            </div>
          ) : null}

          <Link to="/app/account" style={accountButton}>
            Account verwalten
          </Link>

          <DataSourceStatusWidget />
        </motion.div>
      </motion.div>

      <Modal
        isOpen={isHistoryModalOpen}
        onClose={() => setIsHistoryModalOpen(false)}
        title="Vollständige Analyse-Historie"
      >
        {rerunErrorMessage ? (
          <div style={{ ...emptyState, color: theme.colors.dangerText, marginBottom: "12px" }}>
            {rerunErrorMessage}
          </div>
        ) : null}

        <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
          {sortedAnalyses.map((entry) => (
            <div key={entry.job_id} style={{ ...accountRow, alignItems: "flex-start" }}>
              <div>
                <div style={{ ...accountLabel, display: "inline-flex", alignItems: "center", gap: "10px" }}>
                  {entry.symbol} · {HISTORY_MODE_LABELS[entry.mode] ?? entry.mode}
                  <LivePriceBadge symbol={entry.symbol} size="sm" />
                </div>
                <div style={historyTimestamp}>{formatHistoryDate(entry.created_at)}</div>
              </div>

              <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "8px" }}>
                <span style={accountValue}>
                  {HISTORY_STATUS_LABELS[entry.status] ?? entry.status}
                </span>

                {entry.status === "done" ? (
                  <div style={{ display: "flex", gap: "8px" }}>
                    <button
                      type="button"
                      onClick={() => handleViewSnapshot(entry)}
                      style={historyRowButton}
                    >
                      Snapshot anzeigen
                    </button>
                    <button
                      type="button"
                      onClick={() => handleRerun(entry)}
                      style={historyRowButton}
                    >
                      <RotateCcw size={13} />
                      Nochmal analysieren
                    </button>
                  </div>
                ) : null}
              </div>
            </div>
          ))}
        </div>
      </Modal>

      <Modal
        isOpen={snapshotEntry !== null}
        onClose={() => setSnapshotEntry(null)}
        title={
          snapshotEntry
            ? `${snapshotEntry.symbol} · ${formatHistoryDate(snapshotEntry.created_at)}`
            : "Snapshot"
        }
        maxWidth="820px"
      >
        {isLoadingSnapshot ? (
          <div style={emptyState}>Snapshot wird geladen...</div>
        ) : snapshotErrorMessage ? (
          <div style={{ ...emptyState, color: theme.colors.dangerText }}>{snapshotErrorMessage}</div>
        ) : snapshotData?.result_snapshot && snapshotEntry ? (
          <>
            <div style={{ ...historyTimestamp, marginBottom: "16px" }}>
              Historischer Stand vom {formatHistoryDate(snapshotEntry.created_at)}
            </div>
            {snapshotEntry.mode === "custom" ? (
              <CustomAnalysisResultsList
                catalog={snapshotCatalog}
                result={{
                  job_id: snapshotData.job_id,
                  symbol: snapshotData.symbol,
                  status: "done",
                  metrics:
                    (snapshotData.result_snapshot as { metrics?: Record<string, CustomMetricResult> })
                      .metrics ?? {},
                }}
              />
            ) : (
              <AnalyzeResultsDashboard
                data={{
                  job_id: snapshotData.job_id,
                  symbol: snapshotData.symbol,
                  status: "done",
                  total: 0,
                  done: 0,
                  results: snapshotData.result_snapshot as Record<string, CategoryResult>,
                }}
                query=""
              />
            )}
          </>
        ) : (
          <div style={emptyState}>Für diese Analyse ist kein gespeicherter Stand vorhanden.</div>
        )}
      </Modal>
    </div>
  );
}

function formatHistoryDate(value: string): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("de-DE", { dateStyle: "medium", timeStyle: "short" });
}

/* 🎨 STYLES */

const cardLink = {
  textDecoration: "none",
  display: "block",
  width: "100%",
  height: "100%",
};

const mainCardLabel = {
  fontSize: "0.92rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  marginBottom: "14px",
  letterSpacing: "0.02em",
  textTransform: "uppercase" as const,
};

const mainCardTitle = {
  fontSize: "2rem",
  fontWeight: 800,
  marginBottom: "14px",
  lineHeight: 1.2,
  color: theme.colors.textPrimary,
};

const mainCardText = {
  fontSize: "1.16rem",
  lineHeight: 1.8,
  color: theme.colors.textPrimary,
  maxWidth: "820px",
};

const darkCard = {
  background: theme.colors.panel,
  borderRadius: theme.radius.lg,
  padding: "24px",
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
  boxShadow: "0 16px 36px rgba(0, 0, 0, 0.24)",
  cursor: "pointer",
  minHeight: "101px",
  boxSizing: "border-box" as const,
  display: "flex",
  flexDirection: "column" as const,
  justifyContent: "center",
};

const upgradeCard = {
  ...darkCard,
  background: theme.gradients.ctaPrimary,
  border: `1px solid ${theme.colors.chromeBorder}`,
  boxShadow: "0 20px 50px rgba(0, 0, 0, 0.35)",
};

const loadingCard = {
  ...darkCard,
  cursor: "default",
};

const panel = {
  background: theme.glass.subtle.background,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
  borderRadius: theme.radius.lg,
  padding: "28px",
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
  minHeight: "170px",
  boxShadow: "0 14px 34px rgba(0, 0, 0, 0.22)",
  boxSizing: "border-box" as const,
};

const panelTitle = {
  fontSize: "1.42rem",
  fontWeight: 800,
  marginBottom: "18px",
  color: theme.colors.textPrimary,
};

const cardLabel = {
  fontSize: "0.82rem",
  fontWeight: 700,
  color: theme.colors.chrome,
  marginBottom: "10px",
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const cardLabelLight = {
  fontSize: "0.82rem",
  fontWeight: 700,
  color: theme.colors.bgDeep,
  marginBottom: "10px",
  letterSpacing: "0.03em",
  textTransform: "uppercase" as const,
};

const cardTitle = {
  fontSize: "1.32rem",
  fontWeight: 800,
  marginBottom: "8px",
  color: theme.colors.textPrimary,
  lineHeight: 1.35,
};

const cardTitleBright = {
  fontSize: "1.32rem",
  fontWeight: 800,
  marginBottom: "8px",
  color: theme.colors.bgDeep,
  lineHeight: 1.35,
};

const cardText = {
  fontSize: "1.02rem",
  color: theme.colors.textSecondary,
  lineHeight: 1.7,
};

const cardTextBright = {
  fontSize: "1.02rem",
  color: theme.colors.bgDeep,
  lineHeight: 1.7,
};

const emptyState = {
  color: theme.colors.textPrimary,
  fontSize: "1rem",
  lineHeight: 1.8,
};

const historyStackCard = {
  padding: "14px 16px",
  borderRadius: theme.radius.lg,
  background: theme.colors.panel,
  border: `1px solid ${theme.glass.subtle.border}`,
};

const accountRow = {
  display: "flex",
  justifyContent: "space-between",
  marginBottom: "14px",
  alignItems: "center",
};

const accountLabel = {
  color: theme.colors.textSecondary,
  fontSize: "1rem",
};

const accountValue = {
  color: theme.colors.textPrimary,
  fontSize: "1.04rem",
  fontWeight: 800,
};

const historyButton = {
  marginTop: "14px",
  display: "inline-flex",
  alignItems: "center",
  gap: "8px",
  padding: "10px 16px",
  borderRadius: theme.radius.pill,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
  fontWeight: 700,
  fontSize: "0.92rem",
  cursor: "pointer",
};

const historyTimestamp = {
  color: theme.colors.textMuted,
  fontSize: "0.82rem",
  marginTop: "2px",
};

const historyRowButton = {
  display: "inline-flex",
  alignItems: "center",
  gap: "6px",
  padding: "6px 12px",
  borderRadius: theme.radius.pill,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
  fontWeight: 600,
  fontSize: "0.8rem",
  cursor: "pointer",
  whiteSpace: "nowrap" as const,
};

const accountButton = {
  marginTop: "18px",
  display: "block",
  textAlign: "center" as const,
  padding: "14px",
  borderRadius: theme.radius.pill,
  background: theme.colors.panelAlt,
  textDecoration: "none",
  fontWeight: 700,
  fontSize: "1rem",
  color: theme.colors.textPrimary,
};

export default DashboardPage;
