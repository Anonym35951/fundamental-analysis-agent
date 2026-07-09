import { apiRequest } from "./client";

export type SupportCategory =
  | "Allgemeine Frage"
  | "Technisches Problem"
  | "Abrechnung & Abo"
  | "Feedback"
  | "Sonstiges";

export type SupportRequestPayload = {
  category: SupportCategory;
  email: string;
  message: string;
};

/** Oeffentliches Support-Kontaktformular - funktioniert eingeloggt und
 * anonym. apiRequest haengt den Auth-Header nur an, wenn ein Token existiert
 * (siehe client.ts); ohne Token bleibt der Call unauthentifiziert, der
 * Endpunkt gibt fuer anonyme Anfragen nie 401 zurueck. */
export async function sendSupportRequest(payload: SupportRequestPayload): Promise<void> {
  await apiRequest<void>("/support/contact", {
    method: "POST",
    body: payload,
  });
}
