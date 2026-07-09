import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { useLocation } from "react-router-dom";

export type ThemeMode = "light" | "dark";

const STORAGE_KEY = "theme-mode";

type ThemeModeContextValue = {
  mode: ThemeMode;
  toggleMode: () => void;
};

const ThemeModeContext = createContext<ThemeModeContextValue | null>(null);

/** Post-login (authenticated) area always lives under /app/* — see App.tsx.
 * Everything else (landing, login, register, legal pages) is pre-login. */
function isPostLoginPath(pathname: string): boolean {
  return pathname.startsWith("/app");
}

function readStoredMode(): ThemeMode | null {
  const stored = localStorage.getItem(STORAGE_KEY);
  return stored === "dark" || stored === "light" ? stored : null;
}

function getSystemPrefersDark(): boolean {
  return window.matchMedia?.("(prefers-color-scheme: dark)")?.matches ?? false;
}

export function ThemeModeProvider({ children }: { children: ReactNode }) {
  const { pathname } = useLocation();
  const isPostLogin = isPostLoginPath(pathname);

  // Live OS theme preference — kept up to date via a matchMedia listener so
  // pre-login pages (no toggle UI there) track a live OS change, not just
  // the value at first load.
  const [systemPrefersDark, setSystemPrefersDark] = useState(getSystemPrefersDark);

  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    if (!mq) return;
    const handler = (e: MediaQueryListEvent) => setSystemPrefersDark(e.matches);
    mq.addEventListener("change", handler);
    return () => mq.removeEventListener("change", handler);
  }, []);

  // Manually-toggled mode, only ever surfaced/used post-login (the toggle
  // button only exists in the authenticated app shell). Defaults to the
  // system preference on a user's very first-ever visit.
  const [storedMode, setStoredMode] = useState<ThemeMode>(
    () => readStoredMode() ?? (getSystemPrefersDark() ? "dark" : "light"),
  );

  // Pre-login: always the live OS preference, fully independent of whatever
  // the user last chose while logged in. Post-login: the stored manual
  // choice (or system default if never toggled), unaffected by OS changes.
  const mode: ThemeMode = isPostLogin
    ? storedMode
    : systemPrefersDark
      ? "dark"
      : "light";

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", mode);
  }, [mode]);

  useEffect(() => {
    if (isPostLogin) localStorage.setItem(STORAGE_KEY, storedMode);
  }, [isPostLogin, storedMode]);

  const value = useMemo<ThemeModeContextValue>(
    () => ({
      mode,
      toggleMode: () => setStoredMode((prev) => (prev === "light" ? "dark" : "light")),
    }),
    [mode],
  );

  return (
    <ThemeModeContext.Provider value={value}>
      {children}
    </ThemeModeContext.Provider>
  );
}

export function useThemeMode(): ThemeModeContextValue {
  const ctx = useContext(ThemeModeContext);
  if (!ctx) {
    throw new Error("useThemeMode must be used within a ThemeModeProvider");
  }
  return ctx;
}
