import { useEffect, useState } from "react";
import { listCustomers, type CustomerListItem } from "../../api/adminCustomers";
import { theme } from "../ui/theme";
import Input from "../ui/Input";
import { panel, panelTitle, table, tableScroll, th, td, emptyState, trClickable } from "./adminTableStyles";

type CustomerListViewProps = {
  onSelectCustomer: (id: number) => void;
};

const PLAN_OPTIONS = ["", "free", "friends", "pro", "admin"];

function formatName(customer: CustomerListItem) {
  const name = [customer.first_name, customer.last_name].filter(Boolean).join(" ");
  return name || customer.username || "–";
}

function CustomerListView({ onSelectCustomer }: CustomerListViewProps) {
  const [search, setSearch] = useState("");
  const [plan, setPlan] = useState("");
  const [customers, setCustomers] = useState<CustomerListItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    let isCancelled = false;
    // Klassisches Loading-Flag vor einem (hier debounced) Fetch - legitimer
    // Effect-Zweck (LAUNCH_AUDIT.md P2-10).
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setIsLoading(true);

    const timeout = setTimeout(() => {
      listCustomers({ search: search.trim() || undefined, plan: plan || undefined })
        .then((data) => {
          if (!isCancelled) setCustomers(data);
        })
        .catch((error) => {
          if (!isCancelled) {
            setErrorMessage(
              error instanceof Error ? error.message : "Kunden konnten nicht geladen werden."
            );
          }
        })
        .finally(() => {
          if (!isCancelled) setIsLoading(false);
        });
    }, 300);

    return () => {
      isCancelled = true;
      clearTimeout(timeout);
    };
  }, [search, plan]);

  return (
    <section style={panel}>
      <div style={panelTitle}>Kunden</div>

      <div style={{ display: "flex", gap: "12px", flexWrap: "wrap", marginBottom: "18px" }}>
        <Input
          type="text"
          placeholder="Suche nach E-Mail, Benutzername oder Name..."
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          style={{ flex: "1 1 260px" }}
        />
        <select
          value={plan}
          onChange={(event) => setPlan(event.target.value)}
          style={{
            padding: "11px 16px",
            borderRadius: theme.radius.md,
            border: `1px solid ${theme.glass.subtle.border}`,
            background: theme.glass.subtle.background,
            color: theme.colors.textPrimary,
          }}
        >
          {PLAN_OPTIONS.map((option) => (
            <option key={option} value={option}>
              {option === "" ? "Alle Pläne" : option}
            </option>
          ))}
        </select>
      </div>

      {errorMessage ? (
        <div style={{ color: theme.colors.dangerText, marginBottom: "12px" }}>{errorMessage}</div>
      ) : null}

      {isLoading ? (
        <div style={emptyState}>Kunden werden geladen...</div>
      ) : customers.length > 0 ? (
        <div style={tableScroll}>
          <table style={table}>
            <thead>
              <tr>
                <th style={th}>E-Mail</th>
                <th style={th}>Name</th>
                <th style={th}>Plan</th>
                <th style={th}>Status</th>
                <th style={th}>Registriert</th>
              </tr>
            </thead>
            <tbody>
              {customers.map((customer) => (
                <tr
                  key={customer.id}
                  style={trClickable}
                  tabIndex={0}
                  role="button"
                  aria-label={`Kunde ${customer.email} öffnen`}
                  onClick={() => onSelectCustomer(customer.id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelectCustomer(customer.id);
                    }
                  }}
                >
                  <td style={td}>{customer.email}</td>
                  <td style={td}>{formatName(customer)}</td>
                  <td style={td}>{customer.plan}</td>
                  <td style={td}>{customer.billing_status}</td>
                  <td style={td}>
                    {new Date(customer.created_at).toLocaleDateString("de-DE")}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div style={emptyState}>Keine Kunden gefunden.</div>
      )}
    </section>
  );
}

export default CustomerListView;
