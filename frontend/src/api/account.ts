import { apiRequest } from "./client";

export async function changePassword(
  currentPassword: string,
  newPassword: string
): Promise<void> {
  await apiRequest<{ message: string }>("/auth/change-password", {
    method: "PATCH",
    body: {
      current_password: currentPassword,
      new_password: newPassword,
    },
  });
}

type SubscriptionActionResponse = {
  message: string;
  cancel_at_period_end: boolean;
  current_period_end: string | null;
};

export async function cancelSubscription(
  reason?: string
): Promise<SubscriptionActionResponse> {
  return apiRequest<SubscriptionActionResponse>(
    "/billing/cancel-subscription",
    {
      method: "POST",
      body: { reason: reason || null },
    }
  );
}

export type UsageSummary = {
  total_analyses: number;
  distinct_symbols: number;
};

export async function getUsageSummary(): Promise<UsageSummary> {
  return apiRequest<UsageSummary>("/billing/usage-summary");
}

export async function resumeSubscription(): Promise<SubscriptionActionResponse> {
  return apiRequest<SubscriptionActionResponse>(
    "/billing/resume-subscription",
    { method: "POST" }
  );
}

export async function createPortalSession(): Promise<string> {
  const data = await apiRequest<{ url: string }>(
    "/billing/create-portal-session",
    { method: "POST" }
  );

  if (!data?.url) {
    throw new Error("Keine Portal-URL erhalten.");
  }

  return data.url;
}
