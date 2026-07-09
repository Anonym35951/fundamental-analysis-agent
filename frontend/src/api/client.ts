export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;
  /** Machine-readable error code from a structured `detail` object (e.g.
   * "QUOTA_EXCEEDED"), when the backend sends one — undefined for plain
   * string details. */
  code?: string;
  /** Raw structured detail object, for fields beyond message/code (e.g.
   * QUOTA_EXCEEDED's `reset_date`). */
  data?: Record<string, unknown>;

  constructor(message: string, status: number, code?: string, data?: Record<string, unknown>) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.data = data;
  }
}

type RequestOptions = {
  method?: string;
  body?: unknown;
  /** Body is already request-ready (e.g. URLSearchParams) and should not be JSON-encoded. */
  isForm?: boolean;
  /** Skip attaching the Authorization header (for public endpoints like login/register). */
  skipAuth?: boolean;
  headers?: Record<string, string>;
};

function getAuthHeader(): Record<string, string> {
  const accessToken = localStorage.getItem("access_token");
  if (!accessToken) {
    return {};
  }
  const tokenType = localStorage.getItem("token_type") || "bearer";
  return { Authorization: `${tokenType} ${accessToken}` };
}

/** Clears auth tokens and tells any logout-aware state (e.g. CompareProvider's
 * persisted workspace) to reset, via a window event — this module has no
 * React context access, so it can't call hooks directly. Shared by every
 * logout path (explicit button, inactivity timeout, this 401 handler) so
 * none of them can forget to clear stale per-session state. */
export function clearAuthAndWorkspaceState() {
  localStorage.removeItem("access_token");
  localStorage.removeItem("token_type");
  window.dispatchEvent(new Event("app:logout"));
}

function clearSessionAndRedirectToLogin() {
  clearAuthAndWorkspaceState();
  if (window.location.pathname !== "/login") {
    window.location.href = "/login";
  }
}

export async function apiRequest<T>(
  path: string,
  options: RequestOptions = {}
): Promise<T> {
  const { method = "GET", body, isForm = false, skipAuth = false, headers = {} } = options;

  const finalHeaders: Record<string, string> = { ...headers };
  if (!isForm && body !== undefined) {
    finalHeaders["Content-Type"] = "application/json";
  }
  if (!skipAuth) {
    Object.assign(finalHeaders, getAuthHeader());
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    method,
    headers: finalHeaders,
    body: body === undefined ? undefined : isForm ? (body as BodyInit) : JSON.stringify(body),
  });

  if (response.status === 401 && !skipAuth) {
    clearSessionAndRedirectToLogin();
  }

  if (!response.ok) {
    let message = `Anfrage fehlgeschlagen (Status ${response.status}).`;
    let code: string | undefined;
    let data: Record<string, unknown> | undefined;

    try {
      const errorData = await response.json();
      const detail = errorData?.detail;

      if (typeof detail === "string") {
        message = detail;
      } else if (detail && typeof detail === "object") {
        data = detail;
        code = typeof detail.code === "string" ? detail.code : undefined;
        message = typeof detail.message === "string" ? detail.message : JSON.stringify(detail);
      }
    } catch {
      // Fallback auf Standardmeldung
    }

    throw new ApiError(message, response.status, code, data);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}
