import { useState } from "react";
import CustomerListView from "./CustomerListView";
import CustomerDetailView from "./CustomerDetailView";

/** Steuert den Wechsel zwischen Kunden-Liste und Kunden-Detail per lokalem
 * State (kein eigenes Routing) - konsistent mit dem "kein neues Routing"
 * Ansatz des Kunden-Tabs selbst. */
function AdminCustomersTab() {
  const [selectedCustomerId, setSelectedCustomerId] = useState<number | null>(null);

  if (selectedCustomerId !== null) {
    return (
      <CustomerDetailView
        customerId={selectedCustomerId}
        onBack={() => setSelectedCustomerId(null)}
      />
    );
  }

  return <CustomerListView onSelectCustomer={setSelectedCustomerId} />;
}

export default AdminCustomersTab;
