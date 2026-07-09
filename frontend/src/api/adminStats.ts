import { apiRequest } from "./client";

export type FunnelStats = {
  registered: number;
  email_verified: number;
  first_analysis: number;
  five_analyses: number;
  quota_hit: number;
  checkout_started: number;
  subscription_started: number;
};

export type ActivityStats = {
  dau: number;
  wau: number;
  mau: number;
};

export type DailyActivityEntry = {
  date: string;
  registrations: number;
  analyses: number;
};

export type AnalysesBreakdown = {
  by_mode: { mode: string | null; count: number }[];
  top_symbols: { symbol: string | null; count: number }[];
};

export type SubscriptionStats = {
  active_pro_subscriptions: number;
  monthly_subscriptions: number;
  yearly_subscriptions: number;
  mrr_eur: number;
  churned_last_30d: number;
  free_users_near_limit: number;
};

export type NearLimitUser = {
  email: string;
  monthly_request_count: number;
  monthly_request_limit: number;
};

export type ChurnReason = {
  reason: string | null;
  count: number;
};

export function getFunnelStats() {
  return apiRequest<FunnelStats>("/admin/stats/funnel");
}

export function getActivityStats() {
  return apiRequest<ActivityStats>("/admin/stats/activity");
}

export function getDailyActivity(days = 30) {
  return apiRequest<DailyActivityEntry[]>(`/admin/stats/daily-activity?days=${days}`);
}

export function getAnalysesBreakdown() {
  return apiRequest<AnalysesBreakdown>("/admin/stats/analyses");
}

export function getSubscriptionStats() {
  return apiRequest<SubscriptionStats>("/admin/stats/subscriptions");
}

export function getNearLimitUsers() {
  return apiRequest<NearLimitUser[]>("/admin/stats/near-limit-users");
}

export function getChurnReasons() {
  return apiRequest<ChurnReason[]>("/admin/stats/churn-reasons");
}
