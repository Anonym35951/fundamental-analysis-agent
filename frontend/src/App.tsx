import { BrowserRouter, Routes, Route, Navigate, useSearchParams } from "react-router-dom";

import PublicLayout from "./layouts/PublicLayout";
import AuthLayout from "./layouts/AuthLayout";
import AppLayout from "./layouts/AppLayout";

import LandingPage from "./pages/public/LandingPage";

import LoginPage from "./pages/auth/LoginPage";
import RegisterPage from "./pages/auth/RegisterPage";
import ForgotPasswordPage from "./pages/auth/ForgotPasswordPage";
import ResetPasswordPage from "./pages/auth/ResetPasswordPage";
import VerifyEmailPage from "./pages/auth/VerifyEmailPage";

import DashboardPage from "./pages/app/DashBoardPage";
import AnalyzePage from "./pages/app/AnalyzePage";
import ComparePage from "./pages/app/ComparePage";
import BillingPage from "./pages/app/BillingPage";
import AccountPage from "./pages/app/AccountPage";
import SupportPage from "./pages/app/SupportPage";

/** The "eigene Analyse" workflow now lives inside AnalyzePage's
 * "Individuell" tab — this keeps old /app/custom-analysis(?run=<id>)
 * bookmarks/links working by forwarding to the equivalent /app/analyze URL. */
function CustomAnalysisRedirect() {
  const [searchParams] = useSearchParams();
  const runId = searchParams.get("run");
  return <Navigate to={runId ? `/app/analyze?run=${runId}` : "/app/analyze"} replace />;
}

import BillingSuccessPage from "./pages/billing/SuccessPage";
import BillingCancelPage from "./pages/billing/CancelPage";
import AdminDashboardPage from "./pages/app/AdminDashboardPage";

import PrivacyPage from "./pages/legal/PrivacyPage";
import TermsPage from "./pages/legal/TermsPage";
import ImprintPage from "./pages/legal/ImprintPage";
import ContactPage from "./pages/legal/ContactPage";
import CookiesPage from "./pages/legal/CookiesPage";

import ProtectedRoute from "./routes/ProtectedRoute";
import AdminRoute from "./routes/AdminRoute";
import { ThemeModeProvider } from "./components/ui/ThemeModeContext";
import CookieConsentBanner from "./components/consent/CookieConsentBanner";

function App() {
  const token = localStorage.getItem("access_token");

  return (
    <BrowserRouter>
      <ThemeModeProvider>
        <CookieConsentBanner />
        <Routes>
          {/* PUBLIC AREA */}
          <Route element={<PublicLayout />}>
            <Route
              path="/"
              element={
                token ? <Navigate to="/app/dashboard" replace /> : <LandingPage />
              }
            />

            <Route path="/landing" element={<LandingPage />} />

            <Route path="/legal/privacy" element={<PrivacyPage />} />
            <Route path="/legal/terms" element={<TermsPage />} />
            <Route path="/legal/imprint" element={<ImprintPage />} />
            <Route path="/legal/contact" element={<ContactPage />} />
            <Route path="/legal/cookies" element={<CookiesPage />} />
          </Route>

          {/* AUTH AREA */}
          <Route element={<AuthLayout />}>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/register" element={<RegisterPage />} />
            <Route path="/forgot-password" element={<ForgotPasswordPage />} />
            <Route path="/reset-password" element={<ResetPasswordPage />} />
            <Route path="/verify-email" element={<VerifyEmailPage />} />
          </Route>

          {/* PROTECTED APP AREA */}
          <Route element={<ProtectedRoute />}>
            <Route element={<AppLayout />}>
              <Route
                path="/app"
                element={<Navigate to="/app/dashboard" replace />}
              />

              <Route path="/app/dashboard" element={<DashboardPage />} />
              <Route path="/app/analyze" element={<AnalyzePage />} />
              <Route
                path="/app/custom-analysis"
                element={<CustomAnalysisRedirect />}
              />
              <Route path="/app/compare" element={<ComparePage />} />
              <Route path="/app/billing" element={<BillingPage />} />
              <Route path="/app/account" element={<AccountPage />} />
              <Route path="/app/support" element={<SupportPage />} />

              {/* Billing Result Pages jetzt im App-Layout */}
              <Route
                path="/app/billing/success"
                element={<BillingSuccessPage />}
              />
              <Route path="/app/billing/cancel" element={<BillingCancelPage />} />

              {/* Privates Admin-Dashboard — nur für plan=="admin" sichtbar */}
              <Route element={<AdminRoute />}>
                <Route path="/app/admin" element={<AdminDashboardPage />} />
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
    </BrowserRouter>
  );
}

export default App;
