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
import { useFavorites } from "../../hooks/useFavoritesContext";
import { useIsMobile } from "../../hooks/useMediaQuery";
import { theme } from "../ui/theme";
import { useThemeMode } from "../ui/useThemeMode";
import { useToast } from "../ui/useToast";
import { useTranslation } from "../../i18n/useTranslation";
import LivePriceBadge from "../shared/LivePriceBadge";

type AppSidebarProps = {
  onLogout?: () => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
};

type NavKey = "dashboard" | "analyze" | "compare" | "account" | "support" | "billing" | "admin";

const navItems: Array<{ to: string; labelKey: NavKey; icon: typeof LayoutDashboard }> = [
  { to: "/app/dashboard", labelKey: "dashboard", icon: LayoutDashboard },
  { to: "/app/analyze", labelKey: "analyze", icon: LineChart },
  { to: "/app/compare", labelKey: "compare", icon: GitCompare },
  { to: "/app/account", labelKey: "account", icon: UserCircle },
  { to: "/app/support", labelKey: "support", icon: LifeBuoy },
];

// EVOLVING.md EV-080: nur für Free-Nutzer in effectiveNavItems eingefügt -
// Pro/Friends/Admin verwalten ihr Abo über den AccountPage-Portal-Button
// (deckt auch past_due/canceling ab, da diese Zustände weiterhin
// plan==="pro" sind).
const billingNavItem = { to: "/app/billing", labelKey: "billing" as const, icon: CreditCard };

function AppSidebar({
  onLogout,
  isCollapsed,
  onToggleCollapse,
}: AppSidebarProps) {
  const [currentPlan, setCurrentPlan] = useState("free");
  const [isLoadingUser, setIsLoadingUser] = useState(true);
  const { favorites } = useFavorites();
  const [searchQuery, setSearchQuery] = useState("");
  const isMobile = useIsMobile();
  const searchInputRef = useRef<HTMLInputElement | null>(null);
  const { mode, toggleMode } = useThemeMode();
  const { t } = useTranslation("nav");

  const normalizedPlan = currentPlan.trim().toLowerCase();
  const isFreePlan = normalizedPlan === "free";
  const isAdmin = normalizedPlan === "admin";
  const displayPlan = isFreePlan ? t("sidebar.freePlanLabel") : t("sidebar.proPlanLabel");

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

  const effectiveNavItems = useMemo(() => {
    const items = [...navItems];
    // Während des Ladens bleibt der Eintrag ausgeblendet (wie der Admin-
    // Eintrag darunter) - ein kurzes Erscheinen-und-Verschwinden bei Pro/
    // Friends/Admin wäre störender als eine kurze Verzögerung bei Free.
    if (!isLoadingUser && isFreePlan) {
      items.splice(3, 0, billingNavItem);
    }
    if (isAdmin) {
      items.push({ to: "/app/admin", labelKey: "admin" as const, icon: ShieldCheck });
    }
    return items;
  }, [isLoadingUser, isFreePlan, isAdmin]);

  const filteredNavItems = useMemo(
    () =>
      effectiveNavItems.filter((item) =>
        t(`sidebar.${item.labelKey}`).toLowerCase().includes(normalizedQuery)
      ),
    [effectiveNavItems, normalizedQuery, t]
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
        // dvh statt vh: auf iOS Safari folgt die Höhe der tatsächlich
        // sichtbaren Fläche, statt gegen den großen Viewport (Toolbar
        // ausgeblendet) zu rechnen — sonst können Logout/Theme-Button am
        // Boden des Mobile-Drawers unter der Toolbar liegen (RESPONSIVE.md
        // R-P0-6).
        height: "100dvh",
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
        padding: !isMobile && isCollapsed
          ? "18px 12px"
          : isMobile
            // Safe-Area-Puffer für den iOS-Home-Indicator hinter dem
            // Logout-Button (RESPONSIVE.md R-P0-5).
            ? "28px 16px calc(22px + env(safe-area-inset-bottom))"
            : "28px 16px 22px",
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
              aria-label={isCollapsed ? t("sidebar.expandAriaLabel") : t("sidebar.collapseAriaLabel")}
              title={isCollapsed ? t("sidebar.expandTitle") : t("sidebar.collapseTitle")}
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
                placeholder={t("sidebar.searchPlaceholder")}
                style={{
                  width: "100%",
                  height: "42px",
                  paddingLeft: "40px",
                  paddingRight: "36px",
                  borderRadius: theme.radius.md,
                  background: theme.glass.subtle.background,
                  border: `1px solid ${theme.glass.subtle.border}`,
                  color: theme.colors.textPrimary,
                  // >= 16px, sonst zoomt iOS Safari beim Fokussieren die Seite.
                  fontSize: "max(16px, 0.92rem)",
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
                <SectionLabel>{t("sidebar.sectionNavigation")}</SectionLabel>
                <nav style={{ display: "flex", flexDirection: "column", gap: "4px" }}>
                  {filteredNavItems.map((item) => (
                    <SidebarItem key={item.to} to={item.to} label={t(`sidebar.${item.labelKey}`)} icon={item.icon} />
                  ))}
                </nav>
              </div>

              <div data-tour="sidebar-favorites">
                <SectionLabel>{t("sidebar.sectionFavorites")}</SectionLabel>
                {favorites.length === 0 ? (
                  <div
                    style={{
                      color: theme.colors.textMuted,
                      fontSize: "0.86rem",
                      padding: "0 16px",
                      lineHeight: 1.5,
                    }}
                  >
                    {t("sidebar.noFavorites")}
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
              {t("sidebar.statusLabel")}
            </div>

            <div
              style={{
                color: theme.colors.textPrimary,
                fontSize: "1rem",
                fontWeight: 700,
                marginBottom: "6px",
              }}
            >
              {isLoadingUser ? t("sidebar.loading") : displayPlan}
            </div>

            {!isLoadingUser && isFreePlan ? (
              <div
                style={{
                  color: theme.colors.textSecondary,
                  fontSize: "0.92rem",
                  lineHeight: 1.6,
                }}
              >
                {t("sidebar.freePlanHint")}
              </div>
            ) : null}
          </div>

          <motion.button
            onClick={toggleMode}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={theme.motion.spring}
            type="button"
            aria-label={mode === "dark" ? t("sidebar.themeToggleAriaLabelToLight") : t("sidebar.themeToggleAriaLabelToDark")}
            title={mode === "dark" ? t("sidebar.themeLabelLight") : t("sidebar.themeLabelDark")}
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
            {mode === "dark" ? t("sidebar.themeLabelLight") : t("sidebar.themeLabelDark")}
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
            {t("sidebar.logout")}
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
  const { toggleFavorite } = useFavorites();
  const { showToast } = useToast();
  const { t } = useTranslation("nav");

  async function handleRemove(event: React.MouseEvent | React.KeyboardEvent) {
    // Stern sitzt innerhalb des Link-Wrappers (Klick auf die Zeile navigiert
    // zur Analyseseite) - stopPropagation/preventDefault verhindern, dass
    // ein Klick auf den Stern zusätzlich navigiert.
    event.preventDefault();
    event.stopPropagation();
    try {
      await toggleFavorite(symbol);
    } catch {
      showToast(t("sidebar.removeFavoriteError"), "error");
    }
  }

  return (
    <Link
      to={`/app/analyze?symbol=${encodeURIComponent(symbol)}`}
      style={{
        textDecoration: "none",
        display: "flex",
        alignItems: "center",
        gap: "12px",
        minHeight: "44px",
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
      {/* role="button" statt <button>, weil dies innerhalb des Link-Elements
          sitzt - ein verschachteltes <button> in einem <a> ist ungültiges
          HTML (gleiches Idiom wie InfoTooltip.tsx). */}
      <span
        role="button"
        tabIndex={0}
        aria-label={t("sidebar.removeFavoriteAriaLabel", { symbol })}
        onClick={handleRemove}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") handleRemove(event);
        }}
        style={{
          marginLeft: "auto",
          flexShrink: 0,
          display: "flex",
          padding: "4px",
          cursor: "pointer",
        }}
      >
        <Star size={13} color={theme.colors.chrome} fill={theme.colors.chrome} />
      </span>
    </Link>
  );
}

export default AppSidebar;
