import { Link, NavLink } from "react-router-dom";
import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  LayoutDashboard,
  LineChart,
  GitCompare,
  CreditCard,
  UserCircle,
  PanelLeftClose,
  PanelLeftOpen,
  LogOut,
  Search,
  ShieldCheck,
  Star,
  Sun,
  Moon,
  LifeBuoy,
} from "lucide-react";
import { getCurrentUser } from "../../api/auth";
import { getFavorites, type FavoriteEntry } from "../../api/favorites";
import { useIsMobile } from "../../hooks/useMediaQuery";
import { theme } from "../ui/theme";
import { useThemeMode } from "../ui/ThemeModeContext";
import LivePriceBadge from "../shared/LivePriceBadge";

type AppSidebarProps = {
  onLogout?: () => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
};

const navItems = [
  { to: "/app/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/app/analyze", label: "Analyse", icon: LineChart },
  { to: "/app/compare", label: "Vergleich", icon: GitCompare },
  { to: "/app/billing", label: "Billing", icon: CreditCard },
  { to: "/app/account", label: "Account", icon: UserCircle },
  { to: "/app/support", label: "Support", icon: LifeBuoy },
];

function AppSidebar({
  onLogout,
  isCollapsed,
  onToggleCollapse,
}: AppSidebarProps) {
  const [currentPlan, setCurrentPlan] = useState("free");
  const [isLoadingUser, setIsLoadingUser] = useState(true);
  const [favorites, setFavorites] = useState<FavoriteEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const isMobile = useIsMobile();
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const { mode, toggleMode } = useThemeMode();

  const normalizedPlan = currentPlan.trim().toLowerCase();
  const isFreePlan = normalizedPlan === "free";
  const isAdmin = normalizedPlan === "admin";
  const displayPlan = isFreePlan ? "Free Plan" : "Pro Plan";

  useEffect(() => {
    async function loadUser() {
      try {
        const user = await getCurrentUser();
        setCurrentPlan(typeof user.plan === "string" ? user.plan : "free");
      } catch {
        setCurrentPlan("free");
      } finally {
        setIsLoadingUser(false);
      }
    }

    loadUser();
  }, []);

  useEffect(() => {
    getFavorites()
      .then(setFavorites)
      .catch(() => setFavorites([]));
  }, []);

  // "/"-Tastenkürzel fokussiert die Sidebar-Suche, solange der Fokus nicht
  // bereits in einem anderen Eingabefeld liegt — gleiche Konvention wie
  // GitHub/Slack, passend zum Suchfeld-Badge unten.
  useEffect(() => {
    function handleKeyDown(event: KeyboardEvent) {
      if (event.key !== "/") return;
      const activeElement = document.activeElement;
      const isTyping =
        activeElement instanceof HTMLInputElement ||
        activeElement instanceof HTMLTextAreaElement ||
        (activeElement instanceof HTMLElement && activeElement.isContentEditable);
      if (isTyping) return;
      event.preventDefault();
      searchInputRef.current?.focus();
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  const normalizedQuery = searchQuery.trim().toLowerCase();

  const effectiveNavItems = useMemo(
    () =>
      isAdmin
        ? [...navItems, { to: "/app/admin", label: "Admin", icon: ShieldCheck }]
        : navItems,
    [isAdmin]
  );

  const filteredNavItems = useMemo(
    () => effectiveNavItems.filter((item) => item.label.toLowerCase().includes(normalizedQuery)),
    [effectiveNavItems, normalizedQuery]
  );

  const filteredFavorites = useMemo(
    () => favorites.filter((fav) => fav.symbol.toLowerCase().includes(normalizedQuery)),
    [favorites, normalizedQuery]
  );

  return (
    <aside
      style={{
        width: isMobile ? "280px" : isCollapsed ? "76px" : "280px",
        minWidth: isMobile ? "280px" : isCollapsed ? "76px" : "280px",
        height: "100vh",
        position: isMobile ? "fixed" : "sticky",
        top: 0,
        left: 0,
        zIndex: isMobile ? 60 : 1,
        transform: isMobile
          ? isCollapsed
            ? "translateX(-100%)"
            : "translateX(0)"
          : "none",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        background: theme.colors.bgDeep,
        borderRight: `1px solid ${theme.colors.borderSubtle}`,
        padding: !isMobile && isCollapsed ? "18px 12px" : "28px 16px 22px",
        boxSizing: "border-box",
        boxShadow: isMobile
          ? "0 20px 60px rgba(0, 0, 0, 0.5)"
          : "inset -1px 0 0 rgba(255,255,255,0.03)",
        transition:
          "width 0.22s ease, min-width 0.22s ease, padding 0.22s ease, transform 0.22s ease",
        overflow: "hidden",
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", minHeight: 0, flex: 1 }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: isCollapsed ? "center" : "space-between",
            gap: "12px",
            marginBottom: isCollapsed ? "10px" : "22px",
            flexShrink: 0,
          }}
        >
          {!isCollapsed && (
            <NavLink
              to="/app/dashboard"
              style={{
                display: "block",
                textDecoration: "none",
                color: theme.colors.textPrimary,
                minWidth: 0,
                flex: 1,
                paddingLeft: isMobile ? "52px" : 0,
              }}
            >
              <div
                style={{
                  fontSize: "1.7rem",
                  fontWeight: 900,
                  letterSpacing: "-0.05em",
                  lineHeight: 1,
                  whiteSpace: "nowrap",
                }}
                title="ComAnalysis"
              >
                ComAnalysis
              </div>
            </NavLink>
          )}

          {!isMobile && (
            <motion.button
              onClick={onToggleCollapse}
              whileHover={{ scale: 1.06 }}
              whileTap={{ scale: 0.94 }}
              transition={theme.motion.spring}
              type="button"
              aria-label={isCollapsed ? "Sidebar ausklappen" : "Sidebar einklappen"}
              title={isCollapsed ? "Ausklappen" : "Einklappen"}
              style={{
                flexShrink: 0,
                width: "40px",
                height: "40px",
                borderRadius: theme.radius.pill,
                border: `1px solid ${theme.colors.borderSubtle}`,
                background: theme.glass.subtle.background,
                color: theme.colors.textSecondary,
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {isCollapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
            </motion.button>
          )}
        </div>

        {!isCollapsed && (
          <>
            <div style={{ position: "relative", marginBottom: "20px", flexShrink: 0 }}>
              <Search
                size={16}
                color={theme.colors.textMuted}
                style={{ position: "absolute", left: "14px", top: "50%", transform: "translateY(-50%)" }}
              />
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(event) => setSearchQuery(event.target.value)}
                placeholder="Suchen..."
                style={{
                  width: "100%",
                  height: "42px",
                  paddingLeft: "40px",
                  paddingRight: "36px",
                  borderRadius: theme.radius.md,
                  background: theme.glass.subtle.background,
                  border: `1px solid ${theme.glass.subtle.border}`,
                  color: theme.colors.textPrimary,
                  fontSize: "0.92rem",
                  outline: "none",
                  boxSizing: "border-box",
                }}
              />
              {!searchQuery && (
                <span
                  style={{
                    position: "absolute",
                    right: "10px",
                    top: "50%",
                    transform: "translateY(-50%)",
                    padding: "3px 7px",
                    borderRadius: theme.radius.sm,
                    background: theme.colors.panelAlt,
                    border: `1px solid ${theme.colors.borderSubtle}`,
                    color: theme.colors.textMuted,
                    fontSize: "0.74rem",
                    fontWeight: 700,
                    pointerEvents: "none",
                  }}
                >
                  /
                </span>
              )}
            </div>

            <div
              style={{
                flex: 1,
                minHeight: 0,
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                gap: "20px",
              }}
            >
              <div>
                <SectionLabel>Navigation</SectionLabel>
                <nav style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  {filteredNavItems.map((item) => (
                    <SidebarItem key={item.to} to={item.to} label={item.label} icon={item.icon} />
                  ))}
                </nav>
              </div>

              <div data-tour="sidebar-favorites">
                <SectionLabel>Favoriten</SectionLabel>
                {favorites.length === 0 ? (
                  <div
                    style={{
                      color: theme.colors.textMuted,
                      fontSize: "0.86rem",
                      padding: "0 16px",
                      lineHeight: 1.5,
                    }}
                  >
                    Noch keine Favoriten
                  </div>
                ) : (
                  <div style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                    {filteredFavorites.map((fav) => (
                      <FavoriteItem key={fav.symbol} symbol={fav.symbol} />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      {!isCollapsed && (
        <div style={{ flexShrink: 0, marginTop: "16px" }}>
          <div
            style={{
              padding: "16px",
              borderRadius: theme.radius.md,
              background: theme.colors.panel,
              border: `1px solid ${theme.glass.subtle.border}`,
              marginBottom: "14px",
            }}
          >
            <div
              style={{
                fontSize: "0.8rem",
                textTransform: "uppercase",
                letterSpacing: "0.08em",
                color: theme.colors.chrome,
                fontWeight: 700,
                marginBottom: "8px",
              }}
            >
              Status
            </div>

            <div
              style={{
                color: theme.colors.textPrimary,
                fontSize: "1rem",
                fontWeight: 700,
                marginBottom: "6px",
              }}
            >
              {isLoadingUser ? "Lädt..." : displayPlan}
            </div>

            {!isLoadingUser && isFreePlan ? (
              <div
                style={{
                  color: theme.colors.textSecondary,
                  fontSize: "0.92rem",
                  lineHeight: 1.6,
                }}
              >
                Upgrade später dynamisch aus dem Backend.
              </div>
            ) : null}
          </div>

          <motion.button
            onClick={toggleMode}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={theme.motion.spring}
            type="button"
            aria-label={mode === "dark" ? "Hell-Modus aktivieren" : "Dark-Modus aktivieren"}
            title={mode === "dark" ? "Hell-Modus" : "Dark-Modus"}
            style={{
              width: "100%",
              border: `1px solid ${theme.colors.borderSubtle}`,
              borderRadius: theme.radius.pill,
              padding: "13px 16px",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "8px",
              background: theme.glass.subtle.background,
              color: theme.colors.textSecondary,
              fontWeight: 800,
              fontSize: "0.96rem",
              cursor: "pointer",
              marginBottom: "10px",
            }}
          >
            {mode === "dark" ? <Sun size={16} /> : <Moon size={16} />}
            {mode === "dark" ? "Hell-Modus" : "Dark-Modus"}
          </motion.button>

          <motion.button
            onClick={onLogout}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={theme.motion.spring}
            style={{
              width: "100%",
              border: `1px solid ${theme.colors.dangerBorder}`,
              borderRadius: theme.radius.pill,
              padding: "13px 16px",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "8px",
              background: theme.colors.dangerSoft,
              color: theme.colors.dangerText,
              fontWeight: 800,
              fontSize: "0.96rem",
              cursor: "pointer",
            }}
          >
            <LogOut size={16} />
            Logout
          </motion.button>
        </div>
      )}
    </aside>
  );
}

function SectionLabel({ children }: { children: string }) {
  return (
    <div
      style={{
        fontSize: "0.72rem",
        textTransform: "uppercase",
        letterSpacing: "0.08em",
        color: theme.colors.textMuted,
        fontWeight: 700,
        margin: "0 16px 10px",
      }}
    >
      {children}
    </div>
  );
}

type SidebarItemProps = {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
};

function SidebarItem({ to, label, icon: Icon }: SidebarItemProps) {
  return (
    <NavLink
      to={to}
      title={label}
      style={({ isActive }) => ({
        position: "relative",
        textDecoration: "none",
        display: "flex",
        alignItems: "center",
        gap: "12px",
        minHeight: "46px",
        padding: "0 16px",
        borderRadius: theme.radius.pill,
        fontWeight: 700,
        fontSize: "0.95rem",
        color: isActive ? theme.colors.onChrome : theme.colors.textSecondary,
      })}
    >
      {({ isActive }: { isActive: boolean }) => (
        <>
          {isActive && (
            <motion.span
              layoutId="sidebar-active-pill"
              style={{
                position: "absolute",
                inset: 0,
                borderRadius: theme.radius.pill,
                background: theme.gradients.ctaPrimary,
              }}
              transition={theme.motion.spring}
            />
          )}
          <motion.span
            whileHover={isActive ? undefined : { x: 3 }}
            transition={theme.motion.spring}
            style={{
              position: "relative",
              zIndex: 1,
              display: "flex",
              alignItems: "center",
              gap: "12px",
              width: "100%",
            }}
          >
            <Icon size={18} strokeWidth={2} />
            <span>{label}</span>
          </motion.span>
        </>
      )}
    </NavLink>
  );
}

function FavoriteItem({ symbol }: { symbol: string }) {
  return (
    <Link
      to={`/app/analyze?symbol=${encodeURIComponent(symbol)}`}
      style={{
        textDecoration: "none",
        display: "flex",
        alignItems: "center",
        gap: "12px",
        minHeight: "42px",
        padding: "0 16px",
        borderRadius: theme.radius.pill,
        color: theme.colors.textSecondary,
        fontWeight: 700,
        fontSize: "0.9rem",
      }}
    >
      <span
        style={{
          flexShrink: 0,
          width: "26px",
          height: "26px",
          borderRadius: theme.radius.sm,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: theme.colors.chromeSoft,
          border: `1px solid ${theme.colors.chromeBorder}`,
          color: theme.colors.chrome,
          fontSize: "0.68rem",
          fontWeight: 800,
        }}
      >
        {symbol.slice(0, 2)}
      </span>
      <span>{symbol}</span>
      <LivePriceBadge symbol={symbol} size="sm" />
      <Star size={13} style={{ marginLeft: "auto", flexShrink: 0 }} color={theme.colors.chrome} fill={theme.colors.chrome} />
    </Link>
  );
}

export default AppSidebar;
