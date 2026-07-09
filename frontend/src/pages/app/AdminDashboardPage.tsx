import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  getActivityStats,
  getAnalysesBreakdown,
  getChurnReasons,
  getDailyActivity,
  getFunnelStats,
  getNearLimitUsers,
  getSubscriptionStats,
  type ActivityStats,
  type AnalysesBreakdown,
  type ChurnReason,
  type DailyActivityEntry,
  type FunnelStats,
  type NearLimitUser,
  type SubscriptionStats,
} from "../../api/adminStats";
import { theme, useAdminChartTokens } from "../../components/ui/theme";
import AdminCustomersTab from "../../components/admin/AdminCustomersTab";

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  visible: { opacity: 1, y: 0 },
};

const FUNNEL_STAGE_LABELS: { key: keyof FunnelStats; label: string }[] = [
  { key: "registered", label: "Registriert" },
  { key: "email_verified", label: "E-Mail bestätigt" },
  { key: "first_analysis", label: "1. Analyse" },
  { key: "five_analyses", label: "5+ Analysen" },
  { key: "quota_hit", label: "Limit erreicht" },
  { key: "checkout_started", label: "Checkout gestartet" },
  { key: "subscription_started", label: "Zahlender Kunde" },
];

function AdminDashboardPage() {
  const chartTokens = useAdminChartTokens();
  const [activeTab, setActiveTab] = useState<"overview" | "customers">("overview");

  const [funnel, setFunnel] = useState<FunnelStats | null>(null);
  const [activity, setActivity] = useState<ActivityStats | null>(null);
  const [dailyActivity, setDailyActivity] = useState<DailyActivityEntry[]>([]);
  const [analyses, setAnalyses] = useState<AnalysesBreakdown | null>(null);
  const [subscriptions, setSubscriptions] = useState<SubscriptionStats | null>(null);
  const [nearLimitUsers, setNearLimitUsers] = useState<NearLimitUser[]>([]);
  const [churnReasons, setChurnReasons] = useState<ChurnReason[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    async function loadAll() {
      try {
        setErrorMessage("");
        const [
          funnelData,
          activityData,
          dailyActivityData,
          analysesData,
          subscriptionsData,
          nearLimitData,
          churnReasonsData,
        ] = await Promise.all([
          getFunnelStats(),
          getActivityStats(),
          getDailyActivity(30),
          getAnalysesBreakdown(),
          getSubscriptionStats(),
          getNearLimitUsers(),
          getChurnReasons(),
        ]);

        setFunnel(funnelData);
        setActivity(activityData);
        setDailyActivity(dailyActivityData);
        setAnalyses(analysesData);
        setSubscriptions(subscriptionsData);
        setNearLimitUsers(nearLimitData);
        setChurnReasons(churnReasonsData);
      } catch (error) {
        setErrorMessage(
          error instanceof Error ? error.message : "Statistiken konnten nicht geladen werden."
        );
      } finally {
        setIsLoading(false);
      }
    }

    loadAll();
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
        <div style={sectionEyebrow}>Privates Analytics-Dashboard</div>
        <h1 style={pageTitle}>Admin-Übersicht</h1>
        <p style={pageSubtitle}>
          Nur für dich sichtbar. Alle Zahlen kommen aus internen Produkt-Events
          — kein Dritt-Anbieter-Tracking.
        </p>
      </div>

      <div style={tabSwitcher}>
        <button
          type="button"
          onClick={() => setActiveTab("overview")}
          style={tabButton(activeTab === "overview")}
        >
          Übersicht
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("customers")}
          style={tabButton(activeTab === "customers")}
        >
          Kunden
        </button>
      </div>

      {activeTab === "customers" ? <AdminCustomersTab /> : null}

      {activeTab === "overview" && errorMessage ? <div style={errorBox}>{errorMessage}</div> : null}

      {activeTab !== "overview" ? null : isLoading ? (
        <div style={loadingBox}>Statistiken werden geladen...</div>
      ) : (
        <>
          {/* Activity + Subscriptions Kacheln */}
          <div style={statGrid}>
            <StatCard label="DAU" value={activity?.dau} />
            <StatCard label="WAU" value={activity?.wau} />
            <StatCard label="MAU" value={activity?.mau} />
            <StatCard label="Aktive Pro-Abos" value={subscriptions?.active_pro_subscriptions} />
            <StatCard
              label="MRR"
              value={
                subscriptions ? `${subscriptions.mrr_eur.toLocaleString("de-DE")} €` : undefined
              }
            />
            <StatCard label="Churn (30 Tage)" value={subscriptions?.churned_last_30d} />
            <StatCard label="Free nahe am Limit" value={subscriptions?.free_users_near_limit} />
          </div>

          {/* Funnel */}
          <section style={panel}>
            <div style={panelTitle}>Funnel</div>
            <div style={funnelRow}>
              {FUNNEL_STAGE_LABELS.map(({ key, label }) => (
                <div key={key} style={funnelStage}>
                  <div style={funnelValue}>{funnel ? funnel[key] : "–"}</div>
                  <div style={funnelLabel}>{label}</div>
                </div>
              ))}
            </div>
          </section>

          {/* Zeitreihe */}
          <section style={panel}>
            <div style={panelTitle}>Registrierungen &amp; Analysen (30 Tage)</div>
            <div style={{ width: "100%", height: "260px" }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={dailyActivity} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                  <CartesianGrid stroke={chartTokens.grid} strokeDasharray="3 3" vertical={false} />
                  <XAxis dataKey="date" stroke={chartTokens.axis} fontSize={12} />
                  <YAxis stroke={chartTokens.axis} fontSize={12} allowDecimals={false} />
                  <Tooltip
                    contentStyle={{
                      background: theme.colors.panel,
                      border: `1px solid ${theme.glass.subtle.border}`,
                      borderRadius: theme.radius.md,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="registrations"
                    name="Registrierungen"
                    stroke={chartTokens.series[1]}
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="analyses"
                    name="Analysen"
                    stroke={chartTokens.series[4]}
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </section>

          {/* Analysen pro Modus + Top-Symbole */}
          <div style={twoColGrid}>
            <section style={panel}>
              <div style={panelTitle}>Analysen pro Modus</div>
              <div style={{ width: "100%", height: "240px" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={analyses?.by_mode ?? []} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
                    <CartesianGrid stroke={chartTokens.grid} strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="mode" stroke={chartTokens.axis} fontSize={11} />
                    <YAxis stroke={chartTokens.axis} fontSize={12} allowDecimals={false} />
                    <Tooltip
                      contentStyle={{
                        background: theme.colors.panel,
                        border: `1px solid ${theme.glass.subtle.border}`,
                        borderRadius: theme.radius.md,
                      }}
                    />
                    <Bar dataKey="count" name="Analysen" fill={chartTokens.series[2]} radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </section>

            <section style={panel}>
              <div style={panelTitle}>Meistanalysierte Symbole</div>
              {analyses && analyses.top_symbols.length > 0 ? (
                <table style={table}>
                  <thead>
                    <tr>
                      <th style={th}>Symbol</th>
                      <th style={th}>Analysen</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analyses.top_symbols.slice(0, 10).map((row) => (
                      <tr key={row.symbol ?? "unbekannt"}>
                        <td style={td}>{row.symbol ?? "–"}</td>
                        <td style={td}>{row.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={emptyState}>Noch keine Analysen erfasst.</div>
              )}
            </section>
          </div>

          {/* Free-User nahe am Limit + Churn-Gründe */}
          <div style={twoColGrid}>
            <section style={panel}>
              <div style={panelTitle}>Free-User nahe am Limit</div>
              {nearLimitUsers.length > 0 ? (
                <table style={table}>
                  <thead>
                    <tr>
                      <th style={th}>E-Mail</th>
                      <th style={th}>Nutzung</th>
                    </tr>
                  </thead>
                  <tbody>
                    {nearLimitUsers.map((user) => (
                      <tr key={user.email}>
                        <td style={td}>{user.email}</td>
                        <td style={td}>
                          {user.monthly_request_count}/{user.monthly_request_limit}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={emptyState}>Aktuell niemand über 80 % Auslastung.</div>
              )}
            </section>

            <section style={panel}>
              <div style={panelTitle}>Kündigungsgründe</div>
              {churnReasons.length > 0 ? (
                <table style={table}>
                  <thead>
                    <tr>
                      <th style={th}>Grund</th>
                      <th style={th}>Anzahl</th>
                    </tr>
                  </thead>
                  <tbody>
                    {churnReasons.map((row) => (
                      <tr key={row.reason ?? "unbekannt"}>
                        <td style={td}>{row.reason ?? "Kein Grund angegeben"}</td>
                        <td style={td}>{row.count}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <div style={emptyState}>
                  Noch keine erfassten Kündigungsgründe (Erfassung im Cancel-Flow folgt).
                </div>
              )}
            </section>
          </div>
        </>
      )}
    </motion.div>
  );
}

function StatCard({ label, value }: { label: string; value: string | number | undefined }) {
  return (
    <div style={statCard}>
      <div style={statLabel}>{label}</div>
      <div style={statValue}>{value ?? "–"}</div>
    </div>
  );
}

export default AdminDashboardPage;

/* styles */

const pageWrapper = {
  display: "flex",
  flexDirection: "column" as const,
  gap: "24px",
  padding: "10px 6px 20px",
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

const errorBox = {
  padding: "14px 16px",
  borderRadius: theme.radius.md,
  background: theme.colors.dangerSoft,
  border: `1px solid ${theme.colors.dangerBorder}`,
  color: theme.colors.dangerText,
  fontSize: "0.95rem",
};

const loadingBox = {
  padding: "24px",
  color: theme.colors.textSecondary,
  fontSize: "0.98rem",
};

const statGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
  gap: "14px",
};

const statCard = {
  background: theme.glass.subtle.background,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
  borderRadius: theme.radius.lg,
  border: `1px solid ${theme.glass.subtle.border}`,
  padding: "18px 20px",
  boxSizing: "border-box" as const,
};

const statLabel = {
  fontSize: "0.8rem",
  fontWeight: 700,
  color: theme.colors.textSecondary,
  textTransform: "uppercase" as const,
  letterSpacing: "0.04em",
  marginBottom: "8px",
};

const statValue = {
  fontSize: "1.7rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
};

const panel = {
  background: theme.glass.subtle.background,
  backdropFilter: `blur(${theme.glass.subtle.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.subtle.blur})`,
  borderRadius: theme.radius.lg,
  padding: "28px",
  border: `1px solid ${theme.glass.subtle.border}`,
  color: theme.colors.textPrimary,
  boxSizing: "border-box" as const,
};

const panelTitle = {
  fontSize: "1.2rem",
  fontWeight: 800,
  marginBottom: "18px",
  color: theme.colors.textPrimary,
};

const funnelRow = {
  display: "flex",
  flexWrap: "wrap" as const,
  gap: "16px",
};

const funnelStage = {
  flex: "1 1 120px",
  textAlign: "center" as const,
  padding: "14px 10px",
  borderRadius: theme.radius.md,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
};

const funnelValue = {
  fontSize: "1.5rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
};

const funnelLabel = {
  fontSize: "0.78rem",
  color: theme.colors.textSecondary,
  marginTop: "4px",
};

const twoColGrid = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
  gap: "24px",
};

const table = {
  width: "100%",
  borderCollapse: "collapse" as const,
  fontSize: "0.9rem",
};

const th = {
  textAlign: "left" as const,
  padding: "8px 10px",
  color: theme.colors.textSecondary,
  fontWeight: 700,
  fontSize: "0.78rem",
  textTransform: "uppercase" as const,
  letterSpacing: "0.03em",
  borderBottom: `1px solid ${theme.glass.subtle.border}`,
};

const td = {
  padding: "8px 10px",
  color: theme.colors.textPrimary,
  borderBottom: `1px solid ${theme.glass.subtle.border}`,
};

const emptyState = {
  color: theme.colors.textSecondary,
  fontSize: "0.92rem",
  padding: "12px 0",
};

const tabSwitcher = {
  display: "inline-flex",
  gap: "4px",
  padding: "4px",
  borderRadius: theme.radius.pill,
  background: theme.glass.subtle.background,
  border: `1px solid ${theme.glass.subtle.border}`,
};

function tabButton(isActive: boolean) {
  return {
    padding: "7px 16px",
    borderRadius: theme.radius.pill,
    border: "none",
    fontSize: "0.86rem",
    fontWeight: 700,
    cursor: "pointer",
    background: isActive ? theme.colors.chromeStrong : "transparent",
    color: isActive ? theme.colors.onChrome : theme.colors.textSecondary,
    transition: `background ${theme.motion.fast} ${theme.motion.easing}, color ${theme.motion.fast} ${theme.motion.easing}`,
  };
}
