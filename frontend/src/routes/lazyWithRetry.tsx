import { lazy, Suspense, type ComponentType, type JSX, type ReactNode } from "react";
import { theme } from "../components/ui/theme";

const RELOAD_GUARD_KEY = "ev114_chunk_retry_reloaded";

/** Wraps `React.lazy`'s dynamic import so a failed chunk load (typically a
 * stale hash after a Render redeploy replaced dist/assets while this tab
 * still references the old build) triggers exactly one full page reload
 * instead of a permanent broken/blank route. The sessionStorage guard
 * prevents a reload loop if the failure is a genuine, persistent error
 * rather than a stale-chunk 404. */
export function lazyWithRetry<T extends ComponentType<any>>(
  factory: () => Promise<{ default: T }>
) {
  return lazy(async () => {
    try {
      return await factory();
    } catch (error) {
      if (!sessionStorage.getItem(RELOAD_GUARD_KEY)) {
        sessionStorage.setItem(RELOAD_GUARD_KEY, "1");
        window.location.reload();
        // Reload takes over before this would ever resolve/reject again -
        // return a never-settling promise so React doesn't briefly render
        // an error state in the instant before the reload happens.
        return new Promise<never>(() => {});
      }
      throw error;
    }
  });
}

const fallbackStyle = {
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  minHeight: "40vh",
  color: theme.colors.textMuted,
  fontSize: "0.95rem",
};

function RouteLoadingFallback() {
  return <div style={fallbackStyle}>Wird geladen…</div>;
}

/** Suspense boundary scoped to a single route's `element`, not the whole
 * `<Routes>` tree - the surrounding layout (Header/Sidebar/AppLayout) stays
 * mounted while only the route content area shows the fallback during a
 * chunk load. */
export function withSuspense(element: ReactNode): JSX.Element {
  return <Suspense fallback={<RouteLoadingFallback />}>{element}</Suspense>;
}
