import { apiRequest } from "./client";

export type CustomerListItem = {
  id: number;
  email: string;
  username: string | null;
  first_name: string | null;
  last_name: string | null;
  plan: string;
  billing_status: string;
  created_at: string;
  monthly_request_count: number;
  monthly_request_limit: number | null;
};

export type CustomerDetail = CustomerListItem & {
  age: number | null;
  email_verified: boolean;
  current_period_end: string | null;
  stripe_customer_id: string | null;
};

export type CustomerNote = {
  id: number;
  admin_author_id: number | null;
  note: string;
  created_at: string;
};

export type CustomerActivityEntry = {
  id: number;
  event_type: string;
  event_metadata: Record<string, unknown> | null;
  created_at: string;
};

export async function listCustomers(params: {
  search?: string;
  plan?: string;
}): Promise<CustomerListItem[]> {
  const query = new URLSearchParams();
  if (params.search) query.set("search", params.search);
  if (params.plan) query.set("plan", params.plan);
  const queryString = query.toString();
  return apiRequest<CustomerListItem[]>(
    `/admin/customers${queryString ? `?${queryString}` : ""}`
  );
}

export async function getCustomer(id: number): Promise<CustomerDetail> {
  return apiRequest<CustomerDetail>(`/admin/customers/${id}`);
}

export async function listCustomerNotes(id: number): Promise<CustomerNote[]> {
  return apiRequest<CustomerNote[]>(`/admin/customers/${id}/notes`);
}

export async function addCustomerNote(id: number, note: string): Promise<CustomerNote> {
  return apiRequest<CustomerNote>(`/admin/customers/${id}/notes`, {
    method: "POST",
    body: { note },
  });
}

export async function getCustomerActivity(id: number): Promise<CustomerActivityEntry[]> {
  return apiRequest<CustomerActivityEntry[]>(`/admin/customers/${id}/activity`);
}

export async function updateCustomerPlan(
  id: number,
  newPlan: "free" | "friends" | "pro"
): Promise<CustomerDetail> {
  return apiRequest<CustomerDetail>(`/admin/customers/${id}/plan`, {
    method: "POST",
    body: { new_plan: newPlan },
  });
}

export async function resetCustomerUsage(id: number): Promise<CustomerDetail> {
  return apiRequest<CustomerDetail>(`/admin/customers/${id}/reset-usage`, { method: "POST" });
}

export async function deleteCustomer(id: number): Promise<void> {
  await apiRequest<void>(`/admin/customers/${id}`, { method: "DELETE" });
}
