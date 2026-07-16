import { BrowserRouter, Routes, Route, Navigate, useSearchParams } from "react-router-dom";

import PublicLayout from "./layouts/PublicLayout";
import AuthLayout from "./layouts/AuthLayout";
import AppLayout from "./layouts/AppLayout";
import { lazyWithRetry, withSuspense } from "./routes/lazyWithRetry";

// EV-114: alle 22 Seiten waren statisch importiert und landeten dadurch
// zusammen mit recharts/framer-motion/react-joyride in einem einzigen
// ~349-KB-gzip-Haupt-Bundle (EVOLVING.md P1) - jetzt pro Seite per
// lazyWithRetry (React.lazy + Stale-Chunk-Reload-Guard), damit z. B. ein
// Landingpage-Besuch nicht das komplette Dashboard/Analyze/Compare-Chart-
// Bundle mitladen muss. Layouts/Guards/Provider bleiben bewusst statisch.
const LandingPage = lazyWithRetry(() => import("./pages/public/LandingPage"));
const PricingPage = lazyWithRetry(() => import("./pages/public/PricingPage"));

const LoginPage = lazyWithRetry(() => import("./pages/auth/LoginPage"));
const RegisterPage = lazyWithRetry(() => import("./pages/auth/RegisterPage"));
const ForgotPasswordPage = lazyWithRetry(() => import("./pages/auth/ForgotPasswordPage"));
const ResetPasswordPage = lazyWithRetry(() => import("./pages/auth/ResetPasswordPage"));
const VerifyEmailPage = lazyWithRetry(() => import("./pages/auth/VerifyEmailPage"));

const DashboardPage = lazyWithRetry(() => import("./pages/app/DashBoardPage"));
const AnalyzePage = lazyWithRetry(() => import("./pages/app/AnalyzePage"));
const ComparePage = lazyWithRetry(() => import("./pages/app/ComparePage"));
const BillingPage = lazyWithRetry(() => import("./pages/app/BillingPage"));
const AccountPage = lazyWithRetry(() => import("./pages/app/AccountPage"));
const SupportPage = lazyWithRetry(() => import("./pages/app/SupportPage"));

/** The "eigene Analyse" workflow now lives inside AnalyzePage's
 * "Individuell" tab — this keeps old /app/custom-analysis(?run=<id>)
 * bookmarks/links working by forwarding to the equivalent /app/analyze URL. */
function CustomAnalysisRedirect() {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get("run");
  return <Navigate to={runId ? `/app/analyze?run=${runId}` : "/app/analyze"} replace />;
}

const BillingSuccessPage = lazyWithRetry(() => import("./pages/billing/SuccessPage"));
const BillingCancelPage = lazyWithRetry(() => import("./pages/billing/CancelPage"));
const AdminDashboardPage = lazyWithRetry(() => import("./pages/app/AdminDashboardPage"));

const PrivacyPage = lazyWithRetry(() => import("./pages/legal/PrivacyPage"));
const TermsPage = lazyWithRetry(() => import("./pages/legal/TermsPage"));
const ImprintPage = lazyWithRetry(() => import("./pages/legal/ImprintPage"));
const ContactPage = lazyWithRetry(() => import("./pages/legal/ContactPage"));
const CookiesPage = lazyWithRetry(() => import("./pages/legal/CookiesPage"));

import ProtectedRoute from "./routes/ProtectedRoute";
import AdminRoute from "./routes/AdminRoute";
import { ThemeModeProvider } from "./components/ui/ThemeModeContext";
import CookieConsentBanner from "./components/consent/CookieConsentBanner";
import { CompareProvider } from "./hooks/useCompare";
import { AnalysisJobsProvider } from "./hooks/useAnalysisJobs";
import { AnalyzeWorkspaceProvider } from "./hooks/useAnalyzeWorkspace";
import { FavoritesProvider } from "./hooks/useFavorites";

function App() {
  const token = localStorage.getItem("access_token");

  return (
    <BrowserRouter>
      <CompareProvider>
      <AnalysisJobsProvider>
      <AnalyzeWorkspaceProvider>
      <FavoritesProvider>
      <ThemeModeProvider>
        <CookieConsentBanner />
        <Routes>
          {/* PUBLIC AREA */}
          <Route element={<PublicLayout />}>
            <Route
              path="/"
              element={
                token ? <Navigate to="/app/dashboard" replace /> : withSuspense(<LandingPage />)
              }
            />

            <Route path="/landing" element={withSuspense(<LandingPage />)} />
            <Route path="/pricing" element={withSuspense(<PricingPage />)} />

            <Route path="/legal/privacy" element={withSuspense(<PrivacyPage />)} />
            <Route path="/legal/terms" element={withSuspense(<TermsPage />)} />
            <Route path="/legal/imprint" element={withSuspense(<ImprintPage />)} />
            <Route path="/legal/contact" element={withSuspense(<ContactPage />)} />
            <Route path="/legal/cookies" element={withSuspense(<CookiesPage />)} />
          </Route>

          {/* AUTH AREA */}
          <Route element={<AuthLayout />}>
            <Route path="/login" element={withSuspense(<LoginPage />)} />
            <Route path="/register" element={withSuspense(<RegisterPage />)} />
            <Route path="/forgot-password" element={withSuspense(<ForgotPasswordPage />)} />
            <Route path="/reset-password" element={withSuspense(<ResetPasswordPage />)} />
            <Route path="/verify-email" element={withSuspense(<VerifyEmailPage />)} />
          </Route>

          {/* PROTECTED APP AREA */}
          <Route element={<ProtectedRoute />}>
            <Route element={<AppLayout />}>
              <Route
                path="/app"
                element={<Navigate to="/app/dashboard" replace />}
              />

              <Route path="/app/dashboard" element={withSuspense(<DashboardPage />)} />
              <Route path="/app/analyze" element={withSuspense(<AnalyzePage />)} />
              <Route
                path="/app/custom-analysis"
                element={<CustomAnalysisRedirect />}
              />
              <Route path="/app/compare" element={withSuspense(<ComparePage />)} />
              <Route path="/app/billing" element={withSuspense(<BillingPage />)} />
              <Route path="/app/account" element={withSuspense(<AccountPage />)} />
              <Route path="/app/support" element={withSuspense(<SupportPage />)} />

              {/* Billing Result Pages jetzt im App-Layout */}
              <Route
                path="/app/billing/success"
                element={withSuspense(<BillingSuccessPage />)}
              />
              <Route path="/app/billing/cancel" element={withSuspense(<BillingCancelPage />)} />

              {/* Privates Admin-Dashboard — nur für plan=="admin" sichtbar */}
              <Route element={<AdminRoute />}>
                <Route path="/app/admin" element={withSuspense(<AdminDashboardPage />)} />
              </Route>
            </Route>
          </Route>

          {/* Alte Billing-URLs auf neue App-URLs umleiten */}
          <Route
            path="/billing/success"
            element={<Navigate to="/app/billing/success" replace />}
          />
          <Route
            path="/billing/cancel"
            element={<Navigate to="/app/billing/cancel" replace />}
          />

          {/* DEBUG ROUTE (nur in der lokalen Entwicklung) */}
          {import.meta.env.DEV && (
            <Route
              path="/debug"
              element={
                <div style={{ padding: 40 }}>
                  <h1>Debug Route Works</h1>
                  <p>If you see this, routing is working.</p>
                </div>
              }
            />
          )}

          {/* FALLBACK */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </ThemeModeProvider>
      </FavoritesProvider>
      </AnalyzeWorkspaceProvider>
      </AnalysisJobsProvider>
      </CompareProvider>
    </BrowserRouter>
  );
}

export default App;
