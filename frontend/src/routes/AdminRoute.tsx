import { useEffect, useState } from "react";
import { Navigate, Outlet } from "react-router-dom";
import { getCurrentUser } from "../api/auth";

/** UX-Guard fürs private Admin-Dashboard (/app/admin) — versteckt die Seite
 * vor Nicht-Admins. Die eigentliche Zugriffskontrolle liegt serverseitig
 * (require_admin in api/routes/admin_stats.py, plan == "admin"), genau wie
 * bei ProtectedRoute, das nur den Auth-Zustand für die UI spiegelt. */
function AdminRoute() {
  const [status, setStatus] = useState<"checking" | "allowed" | "denied">("checking");

  useEffect(() => {
    let isMounted = true;

    getCurrentUser()
      .then((user) => {
        if (isMounted) setStatus(user.plan === "admin" ? "allowed" : "denied");
      })
      .catch(() => {
        if (isMounted) setStatus("denied");
      });

    return () => {
      isMounted = false;
    };
  }, []);

  if (status === "checking") return null;
  if (status === "denied") return <Navigate to="/app/dashboard" replace />;

  return <Outlet />;
}

export default AdminRoute;
