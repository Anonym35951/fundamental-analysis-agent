import { apiRequest } from "./client";

type BillingInterval = "month" | "year";

type CreateCheckoutSessionResponse = {
  checkout_url: string;
};

export async function createCheckoutSession(
  billingInterval: BillingInterval
): Promise<string> {
  const data = await apiRequest<CreateCheckoutSessionResponse>(
    "/billing/create-checkout-session",
    {
      method: "POST",
      body: { billing_interval: billingInterval },
    }
  );

  if (!data.checkout_url) {
    throw new Error("Keine Checkout-URL vom Server erhalten.");
  }

  return data.checkout_url;
}
