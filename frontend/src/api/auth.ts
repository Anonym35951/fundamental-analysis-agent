import { apiRequest } from "./client";

export type RegisterPayload = {
  email: string;
  password: string;
  username: string;
  first_name: string;
  last_name: string;
  // ISO "YYYY-MM-DD" - ersetzt "age" (aendert sich nicht jaehrlich, siehe
  // api/models/user.py).
  birth_date: string;
  terms_accepted: boolean;
  privacy_accepted: boolean;
};

export type UserProfileUpdatePayload = Partial<{
  username: string;
  first_name: string;
  last_name: string;
  birth_date: string;
}>;

export type LoginPayload = {
  username: string;
  password: string;
};

export type LoginResponse = {
  access_token: string;
  token_type: string;
};

export type CurrentUserResponse = {
  id: number;
  email: string;
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  email_verified: boolean;
  plan: string;
  billing_status: string;
  billing_interval: string | null;
  current_period_end: string | null;
  grace_until: string | null;
  monthly_request_count: number;
  monthly_request_limit: number | null;
  stripe_customer_id: string | null;
  stripe_subscription_id: string | null;
  stripe_price_id: string | null;

  username: string | null;
  first_name: string | null;
  last_name: string | null;
  // Legacy: nur bei vor der Umstellung auf birth_date registrierten Konten
  // gesetzt. birth_date ist der neue, kanonische Wert.
  age: number | null;
  birth_date: string | null;
  onboarding_completed_at: string | null;
};

export async function registerUser(payload: RegisterPayload): Promise<void> {
  await apiRequest<void>("/auth/register", {
    method: "POST",
    body: payload,
    skipAuth: true,
  });
}

export async function loginUser(
  payload: LoginPayload
): Promise<LoginResponse> {
  const formData = new URLSearchParams();
  formData.append("username", payload.username);
  formData.append("password", payload.password);

  return apiRequest<LoginResponse>("/auth/login", {
    method: "POST",
    body: formData,
    isForm: true,
    skipAuth: true,
  });
}

export async function getCurrentUser(): Promise<CurrentUserResponse> {
  if (!localStorage.getItem("access_token")) {
    throw new Error("Nicht eingeloggt.");
  }

  return apiRequest<CurrentUserResponse>("/auth/me");
}

/** Optionale Profil-Nachpflege (Benutzername/Vorname/Nachname/Alter) -
 * fuer Bestandsnutzer freiwillig ueber die Account-Seite, kein Zwang. */
export async function updateProfile(
  payload: UserProfileUpdatePayload
): Promise<CurrentUserResponse> {
  return apiRequest<CurrentUserResponse>("/auth/profile", {
    method: "PATCH",
    body: payload,
  });
}

/** Markiert die gefuehrte Onboarding-Tour als abgeschlossen (oder
 * uebersprungen), damit sie beim naechsten Login nicht erneut startet. */
export async function completeOnboarding(): Promise<void> {
  await apiRequest<void>("/auth/onboarding-complete", { method: "POST" });
}

/** Exchanges the current (still-valid) token for a fresh one — used for the
 * session keep-alive background renewal and the "Sitzung verlängern"
 * inactivity-warning flow in SessionTimeoutWatcher. */
export async function refreshToken(): Promise<LoginResponse> {
  return apiRequest<LoginResponse>("/auth/refresh", { method: "POST" });
}

export async function requestPasswordReset(email: string): Promise<void> {
  await apiRequest<void>("/auth/forgot-password", {
    method: "POST",
    body: { email },
    skipAuth: true,
  });
}

export async function resetPassword(
  token: string,
  newPassword: string
): Promise<void> {
  await apiRequest<void>("/auth/reset-password", {
    method: "POST",
    body: { token, new_password: newPassword },
    skipAuth: true,
  });
}

export async function verifyEmail(token: string): Promise<void> {
  await apiRequest<void>("/auth/verify-email", {
    method: "POST",
    body: { token },
    skipAuth: true,
  });
}

/** Fordert eine neue Bestätigungs-Mail an (eingeloggt, aber unverifiziert). */
export async function resendVerificationEmail(): Promise<void> {
  await apiRequest<void>("/auth/resend-verification", {
    method: "POST",
  });
}

/** Gegenstück zu resendVerificationEmail für den Fall, dass der Login selbst
 * wegen fehlender Verifizierung blockiert ist (kein Token verfügbar). Nimmt
 * dieselbe Email/Benutzername-Eingabe wie das Login-Formular entgegen. */
export async function resendVerificationEmailPublic(identifier: string): Promise<void> {
  await apiRequest<void>("/auth/resend-verification-public", {
    method: "POST",
    body: { identifier },
    skipAuth: true,
  });
}

/** DSGVO-Konto-Löschung: kündigt ein laufendes Abo sofort und löscht alle
 * Nutzerdaten endgültig. Erfordert das aktuelle Passwort zur Bestätigung. */
export async function deleteAccount(password: string): Promise<void> {
  await apiRequest<void>("/auth/delete-account", {
    method: "POST",
    body: { password },
  });
}
