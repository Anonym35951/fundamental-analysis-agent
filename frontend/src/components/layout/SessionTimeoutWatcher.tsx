import { useCallback, useEffect, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { refreshToken } from "../../api/auth";
import { clearAuthAndWorkspaceState } from "../../api/client";
import Modal from "../ui/Modal";
import Button from "../ui/Button";
import { theme } from "../ui/theme";

const KEEP_ALIVE_INTERVAL_MS = 10 * 60 * 1000;
const IDLE_WARNING_MS = 30 * 60 * 1000;
const COUNTDOWN_MS = 2 * 60 * 1000;
const ACTIVITY_EVENTS = ["mousemove", "keydown", "mousedown", "wheel", "touchstart"] as const;

function formatCountdown(ms: number): string {
  const totalSeconds = Math.max(0, Math.ceil(ms / 1000));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

/** Mounted once in AppLayout, active across every authenticated page.
 * Three independent timers:
 *  - a keep-alive interval that silently renews the backend token every 10
 *    minutes so its fixed-at-issuance expiry never causes a surprise logout
 *    while the app is actually open;
 *  - an idle timer, reset only by real user input (mouse/keyboard/touch),
 *    that opens the "extend session?" warning after 30 minutes of true
 *    inactivity — background polling elsewhere in the app must NOT reset
 *    this, only genuine interaction counts;
 *  - a 2-minute countdown that logs the user out if the warning goes
 *    unanswered, leaving a sticky message for the login page to show. */
export default function SessionTimeoutWatcher() {
  const navigate = useNavigate();
  const [isWarningOpen, setIsWarningOpen] = useState(false);
  const [remainingMs, setRemainingMs] = useState(COUNTDOWN_MS);

  const idleTimeoutRef = useRef<number | null>(null);
  const countdownIntervalRef = useRef<number | null>(null);
  const isWarningOpenRef = useRef(false);

  useEffect(() => {
    isWarningOpenRef.current = isWarningOpen;
  }, [isWarningOpen]);

  const performLogout = useCallback(() => {
    if (countdownIntervalRef.current) window.clearInterval(countdownIntervalRef.current);
    clearAuthAndWorkspaceState();
    sessionStorage.setItem("logged_out_reason", "inactivity");
    setIsWarningOpen(false);
    navigate("/login");
  }, [navigate]);

  const openWarning = useCallback(() => {
    setRemainingMs(COUNTDOWN_MS);
    setIsWarningOpen(true);
    const deadline = Date.now() + COUNTDOWN_MS;
    countdownIntervalRef.current = window.setInterval(() => {
      const left = deadline - Date.now();
      if (left <= 0) {
        performLogout();
        return;
      }
      setRemainingMs(left);
    }, 1000);
  }, [performLogout]);

  const scheduleIdleTimer = useCallback(() => {
    if (idleTimeoutRef.current) window.clearTimeout(idleTimeoutRef.current);
    idleTimeoutRef.current = window.setTimeout(openWarning, IDLE_WARNING_MS);
  }, [openWarning]);

  const handleExtend = useCallback(() => {
    if (countdownIntervalRef.current) window.clearInterval(countdownIntervalRef.current);
    setIsWarningOpen(false);
    refreshToken()
      .then((response) => {
        localStorage.setItem("access_token", response.access_token);
        localStorage.setItem("token_type", response.token_type);
      })
      .catch(() => {
        // The existing global 401 handler in api/client.ts already redirects
        // to /login if the underlying token had actually expired.
      });
    scheduleIdleTimer();
  }, [scheduleIdleTimer]);

  useEffect(() => {
    refreshToken()
      .then((response) => {
        localStorage.setItem("access_token", response.access_token);
        localStorage.setItem("token_type", response.token_type);
      })
      .catch(() => {});

    const keepAliveId = window.setInterval(() => {
      refreshToken()
        .then((response) => {
          localStorage.setItem("access_token", response.access_token);
          localStorage.setItem("token_type", response.token_type);
        })
        .catch(() => {});
    }, KEEP_ALIVE_INTERVAL_MS);

    scheduleIdleTimer();

    function handleActivity() {
      if (isWarningOpenRef.current) return;
      scheduleIdleTimer();
    }

    ACTIVITY_EVENTS.forEach((eventName) => window.addEventListener(eventName, handleActivity, { passive: true }));

    return () => {
      window.clearInterval(keepAliveId);
      if (idleTimeoutRef.current) window.clearTimeout(idleTimeoutRef.current);
      if (countdownIntervalRef.current) window.clearInterval(countdownIntervalRef.current);
      ACTIVITY_EVENTS.forEach((eventName) => window.removeEventListener(eventName, handleActivity));
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <Modal isOpen={isWarningOpen} onClose={handleExtend} title="Sitzung läuft bald ab">
      <p style={{ margin: "0 0 18px 0", color: theme.colors.textSecondary, fontSize: "0.98rem", lineHeight: 1.7 }}>
        Du warst länger nicht aktiv. Aus Sicherheitsgründen wirst du in{" "}
        <strong style={{ color: theme.colors.textPrimary }}>{formatCountdown(remainingMs)}</strong> automatisch
        abgemeldet, falls du nicht reagierst.
      </p>
      <Button variant="cta" onClick={handleExtend} style={{ width: "100%" }}>
        Sitzung verlängern
      </Button>
    </Modal>
  );
}
