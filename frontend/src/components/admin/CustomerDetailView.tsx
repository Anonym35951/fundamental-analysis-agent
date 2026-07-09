import { useEffect, useState } from "react";
import { ArrowLeft } from "lucide-react";
import {
  addCustomerNote,
  getCustomer,
  getCustomerActivity,
  listCustomerNotes,
  resetCustomerUsage,
  updateCustomerPlan,
  type CustomerActivityEntry,
  type CustomerDetail as CustomerDetailType,
  type CustomerNote,
} from "../../api/adminCustomers";
import { theme } from "../ui/theme";
import Button from "../ui/Button";
import Modal from "../ui/Modal";
import { panel, panelTitle, emptyState } from "./adminTableStyles";

type CustomerDetailViewProps = {
  customerId: number;
  onBack: () => void;
};

function formatDateTime(value: string) {
  return new Date(value).toLocaleString("de-DE");
}

function CustomerDetailView({ customerId, onBack }: CustomerDetailViewProps) {
  const [customer, setCustomer] = useState<CustomerDetailType | null>(null);
  const [notes, setNotes] = useState<CustomerNote[]>([]);
  const [activity, setActivity] = useState<CustomerActivityEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  const [newNote, setNewNote] = useState("");
  const [isSavingNote, setIsSavingNote] = useState(false);

  const [selectedPlan, setSelectedPlan] = useState<"free" | "friends" | "pro">("free");
  const [isPlanModalOpen, setIsPlanModalOpen] = useState(false);
  const [isResetModalOpen, setIsResetModalOpen] = useState(false);
  const [isSavingPlan, setIsSavingPlan] = useState(false);
  const [isResettingUsage, setIsResettingUsage] = useState(false);

  useEffect(() => {
    let isCancelled = false;
    setIsLoading(true);
    setErrorMessage("");

    Promise.all([
      getCustomer(customerId),
      listCustomerNotes(customerId),
      getCustomerActivity(customerId),
    ])
      .then(([customerData, notesData, activityData]) => {
        if (isCancelled) return;
        setCustomer(customerData);
        setNotes(notesData);
        setActivity(activityData);
        // "admin" darf hier nie selektierbar sein (siehe Dropdown unten) -
        // Guard faengt den unerwarteten Fall ab, dass ein Kunde bereits
        // plan="admin" hat (z.B. ueber scripts/set_admin.py gesetzt).
        setSelectedPlan(customerData.plan === "free" || customerData.plan === "friends" || customerData.plan === "pro" ? customerData.plan : "free");
      })
      .catch((error) => {
        if (!isCancelled) {
          setErrorMessage(
            error instanceof Error ? error.message : "Kundendaten konnten nicht geladen werden."
          );
        }
      })
      .finally(() => {
        if (!isCancelled) setIsLoading(false);
      });

    return () => {
      isCancelled = true;
    };
  }, [customerId]);

  async function handleAddNote() {
    if (!newNote.trim() || isSavingNote) return;
    setIsSavingNote(true);
    try {
      const created = await addCustomerNote(customerId, newNote.trim());
      setNotes((prev) => [created, ...prev]);
      setNewNote("");
    } catch (error) {
      setErrorMessage(
        error instanceof Error ? error.message : "Notiz konnte nicht gespeichert werden."
      );
    } finally {
      setIsSavingNote(false);
    }
  }

  async function handlePlanChange() {
    if (isSavingPlan) return;
    setIsSavingPlan(true);
    try {
      const updated = await updateCustomerPlan(customerId, selectedPlan);
      setCustomer(updated);
      setIsPlanModalOpen(false);
      setNotes(await listCustomerNotes(customerId));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Plan konnte nicht geändert werden.");
    } finally {
      setIsSavingPlan(false);
    }
  }

  async function handleResetUsage() {
    if (isResettingUsage) return;
    setIsResettingUsage(true);
    try {
      const updated = await resetCustomerUsage(customerId);
      setCustomer(updated);
      setIsResetModalOpen(false);
      setNotes(await listCustomerNotes(customerId));
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "Kontingent konnte nicht zurückgesetzt werden.");
    } finally {
      setIsResettingUsage(false);
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
      <button
        type="button"
        onClick={onBack}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: "8px",
          background: "transparent",
          border: "none",
          color: theme.colors.textSecondary,
          fontWeight: 600,
          cursor: "pointer",
          padding: 0,
          width: "fit-content",
        }}
      >
        <ArrowLeft size={16} />
        Zurück zur Kundenliste
      </button>

      {errorMessage ? (
        <div style={{ color: theme.colors.dangerText }}>{errorMessage}</div>
      ) : null}

      {isLoading ? (
        <div style={emptyState}>Kundendaten werden geladen...</div>
      ) : customer ? (
        <>
          <section style={panel}>
            <div style={panelTitle}>Profil</div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
                gap: "14px",
              }}
            >
              <DetailField label="E-Mail" value={customer.email} />
              <DetailField label="Benutzername" value={customer.username ?? "–"} />
              <DetailField
                label="Name"
                value={
                  [customer.first_name, customer.last_name].filter(Boolean).join(" ") || "–"
                }
              />
              <DetailField label="Alter" value={customer.age ?? "–"} />
              <DetailField label="Plan" value={customer.plan} />
              <DetailField label="Billing-Status" value={customer.billing_status} />
              <DetailField
                label="E-Mail bestätigt"
                value={customer.email_verified ? "Ja" : "Nein"}
              />
              <DetailField
                label="Nutzung"
                value={`${customer.monthly_request_count}/${customer.monthly_request_limit ?? "∞"}`}
              />
              <DetailField
                label="Registriert am"
                value={new Date(customer.created_at).toLocaleDateString("de-DE")}
              />
            </div>
          </section>

          <section style={panel}>
            <div style={panelTitle}>Admin-Aktionen</div>
            <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
              <div>
                <div style={adminActionLabel}>Plan ändern</div>
                <p style={adminActionHint}>
                  Setzt den Plan manuell. „Friends" vergibt unbegrenztes Kontingent ohne
                  echtes Stripe-Abo und taucht daher nicht in der Abo-/MRR-Auswertung auf.
                </p>
                <div style={{ display: "flex", gap: "10px", flexWrap: "wrap", alignItems: "center" }}>
                  <select
                    value={selectedPlan}
                    onChange={(event) => setSelectedPlan(event.target.value as "free" | "friends" | "pro")}
                    style={selectStyle}
                  >
                    <option value="free">Free</option>
                    <option value="friends">Friends (unbegrenzt, geschenkt)</option>
                    <option value="pro">Pro (echtes Stripe-Abo)</option>
                  </select>
                  <Button
                    variant="secondary"
                    onClick={() => setIsPlanModalOpen(true)}
                    disabled={selectedPlan === customer.plan}
                  >
                    Plan ändern
                  </Button>
                </div>
              </div>
              <div>
                <div style={adminActionLabel}>Verbrauchskontingent</div>
                <p style={adminActionHint}>
                  Setzt den monatlichen Nutzungszähler dieses Kunden auf 0 zurück. Das
                  Kontingent-Limit selbst bleibt unverändert.
                </p>
                <Button variant="secondary" onClick={() => setIsResetModalOpen(true)}>
                  Kontingent zurücksetzen
                </Button>
              </div>
            </div>
          </section>

          <Modal
            isOpen={isPlanModalOpen}
            onClose={() => setIsPlanModalOpen(false)}
            title="Plan wirklich ändern?"
            maxWidth="480px"
          >
            <p style={confirmMessageStyle}>
              Plan von <strong>{customer.plan}</strong> zu <strong>{selectedPlan}</strong> ändern für{" "}
              {customer.email}? Diese Änderung wird automatisch als Notiz protokolliert.
            </p>
            <div style={confirmButtonRow}>
              <Button variant="ghost" onClick={() => setIsPlanModalOpen(false)} disabled={isSavingPlan}>
                Abbrechen
              </Button>
              <Button variant="primary" onClick={handlePlanChange} disabled={isSavingPlan}>
                {isSavingPlan ? "Wird geändert..." : "Plan ändern"}
              </Button>
            </div>
          </Modal>

          <Modal
            isOpen={isResetModalOpen}
            onClose={() => setIsResetModalOpen(false)}
            title="Kontingent wirklich zurücksetzen?"
            maxWidth="480px"
          >
            <p style={confirmMessageStyle}>
              Der monatliche Nutzungszähler von {customer.email} wird auf 0 gesetzt. Diese
              Änderung wird automatisch als Notiz protokolliert.
            </p>
            <div style={confirmButtonRow}>
              <Button variant="ghost" onClick={() => setIsResetModalOpen(false)} disabled={isResettingUsage}>
                Abbrechen
              </Button>
              <Button variant="primary" onClick={handleResetUsage} disabled={isResettingUsage}>
                {isResettingUsage ? "Wird zurückgesetzt..." : "Zurücksetzen"}
              </Button>
            </div>
          </Modal>

          <section style={panel}>
            <div style={panelTitle}>Notizen</div>

            <div style={{ display: "flex", gap: "10px", marginBottom: "18px", flexWrap: "wrap" }}>
              <textarea
                value={newNote}
                onChange={(event) => setNewNote(event.target.value)}
                placeholder="Neue Notiz hinzufügen..."
                rows={2}
                style={{
                  flex: "1 1 260px",
                  padding: "11px 14px",
                  borderRadius: theme.radius.md,
                  border: `1px solid ${theme.glass.subtle.border}`,
                  background: theme.glass.subtle.background,
                  color: theme.colors.textPrimary,
                  fontFamily: "inherit",
                  fontSize: "0.95rem",
                  resize: "vertical",
                }}
              />
              <Button
                variant="primary"
                onClick={handleAddNote}
                disabled={isSavingNote || !newNote.trim()}
              >
                {isSavingNote ? "Speichert..." : "Notiz speichern"}
              </Button>
            </div>

            {notes.length > 0 ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                {notes.map((note) => (
                  <div
                    key={note.id}
                    style={{
                      padding: "12px 14px",
                      borderRadius: theme.radius.md,
                      background: theme.glass.subtle.background,
                      border: `1px solid ${theme.glass.subtle.border}`,
                    }}
                  >
                    <div style={{ color: theme.colors.textPrimary, marginBottom: "6px" }}>
                      {note.note}
                    </div>
                    <div style={{ color: theme.colors.textMuted, fontSize: "0.78rem" }}>
                      {formatDateTime(note.created_at)}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div style={emptyState}>Noch keine Notizen für diesen Kunden.</div>
            )}
          </section>

          <section style={panel}>
            <div style={panelTitle}>Aktivität</div>
            {activity.length > 0 ? (
              <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                {activity.map((entry) => (
                  <div
                    key={entry.id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: "12px",
                      padding: "8px 0",
                      borderBottom: `1px solid ${theme.glass.subtle.border}`,
                      fontSize: "0.9rem",
                    }}
                  >
                    <span style={{ color: theme.colors.textPrimary }}>{entry.event_type}</span>
                    <span style={{ color: theme.colors.textMuted }}>
                      {formatDateTime(entry.created_at)}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div style={emptyState}>Noch keine erfasste Aktivität.</div>
            )}
          </section>
        </>
      ) : null}
    </div>
  );
}

function DetailField({ label, value }: { label: string; value: string | number }) {
  return (
    <div>
      <div
        style={{
          fontSize: "0.78rem",
          fontWeight: 700,
          color: theme.colors.textSecondary,
          textTransform: "uppercase",
          letterSpacing: "0.03em",
          marginBottom: "4px",
        }}
      >
        {label}
      </div>
      <div style={{ color: theme.colors.textPrimary }}>{value}</div>
    </div>
  );
}

const adminActionLabel: React.CSSProperties = {
  fontSize: "0.92rem",
  fontWeight: 700,
  color: theme.colors.textPrimary,
  marginBottom: "6px",
};

const adminActionHint: React.CSSProperties = {
  margin: "0 0 12px 0",
  color: theme.colors.textSecondary,
  fontSize: "0.85rem",
  lineHeight: 1.6,
};

const selectStyle: React.CSSProperties = {
  padding: "11px 14px",
  borderRadius: theme.radius.md,
  border: `1px solid ${theme.glass.subtle.border}`,
  background: theme.glass.subtle.background,
  color: theme.colors.textPrimary,
  fontFamily: "inherit",
  fontSize: "0.95rem",
};

const confirmMessageStyle: React.CSSProperties = {
  color: theme.colors.textPrimary,
  fontSize: "0.95rem",
  lineHeight: 1.7,
  margin: "0 0 20px 0",
};

const confirmButtonRow: React.CSSProperties = {
  display: "flex",
  justifyContent: "flex-end",
  gap: "10px",
};

export default CustomerDetailView;
