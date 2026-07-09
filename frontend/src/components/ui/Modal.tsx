import type { ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { X } from "lucide-react";
import { theme } from "./theme";

type ModalProps = {
  isOpen: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  maxWidth?: string;
};

/** Generic backdrop-fade + scale-in modal shell, factored out of the
 * confirm-dialog pattern in ConfirmModal.tsx so new modals (history list,
 * future dialogs) reuse the same entrance animation and glass surface
 * instead of re-implementing it per usage. */
export default function Modal({ isOpen, onClose, title, children, maxWidth = "560px" }: ModalProps) {
  return (
    <AnimatePresence>
      {isOpen ? (
        <motion.div
          style={overlayStyle}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
          onClick={onClose}
        >
          <motion.div
            style={{ ...modalStyle, maxWidth }}
            initial={{ opacity: 0, scale: 0.94, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 6 }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
            onClick={(event) => event.stopPropagation()}
          >
            <div style={headerRow}>
              <h2 style={titleStyle}>{title}</h2>
              <button onClick={onClose} aria-label="Schließen" style={closeButton}>
                <X size={18} />
              </button>
            </div>

            <div style={bodyStyle}>{children}</div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

const overlayStyle: React.CSSProperties = {
  position: "fixed",
  top: 0,
  left: 0,
  width: "100vw",
  height: "100vh",
  background: "rgba(0, 0, 0, 0.78)",
  backdropFilter: "blur(6px)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 1000,
  padding: "20px",
};

const modalStyle: React.CSSProperties = {
  width: "100%",
  // Solid (not glass) panel color: the modal sits on top of a deliberately
  // dark overlay scrim (see overlayStyle below), and the near-transparent
  // glass tokens would let that dark scrim show through regardless of
  // theme mode — theme.colors.panel is opaque enough to stay readable in
  // both modes instead.
  background: theme.colors.panel,
  backdropFilter: `blur(${theme.glass.elevated.blur})`,
  WebkitBackdropFilter: `blur(${theme.glass.elevated.blur})`,
  borderRadius: theme.radius.lg,
  padding: "26px 24px 22px",
  border: `1px solid ${theme.glass.elevated.border}`,
  boxShadow: theme.glass.elevated.shadow,
  maxHeight: "80vh",
  display: "flex",
  flexDirection: "column",
};

const headerRow: React.CSSProperties = {
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  marginBottom: "16px",
  gap: "12px",
};

const titleStyle: React.CSSProperties = {
  margin: 0,
  fontSize: "1.4rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
};

const closeButton: React.CSSProperties = {
  flexShrink: 0,
  width: "32px",
  height: "32px",
  borderRadius: theme.radius.pill,
  border: `1px solid ${theme.glass.subtle.border}`,
  background: theme.glass.subtle.background,
  color: theme.colors.textSecondary,
  cursor: "pointer",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
};

const bodyStyle: React.CSSProperties = {
  overflowY: "auto",
};
