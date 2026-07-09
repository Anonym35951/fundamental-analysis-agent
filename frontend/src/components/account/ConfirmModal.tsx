import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { theme } from "../ui/theme";

type ConfirmModalProps = {
  isOpen: boolean;
  title: string;
  message: string;
  confirmText?: string;
  cancelText?: string;
  isLoading?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
};

function ConfirmModal({
  isOpen,
  title,
  message,
  confirmText = "Bestätigen",
  cancelText = "Abbrechen",
  isLoading = false,
  onConfirm,
  onCancel,
}: ConfirmModalProps) {
  return (
    <AnimatePresence>
      {isOpen ? (
        <motion.div
          style={overlayStyle}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.18 }}
        >
          <motion.div
            style={modalStyle}
            initial={{ opacity: 0, scale: 0.94, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 6 }}
            transition={{ duration: 0.2, ease: [0.4, 0, 0.2, 1] }}
          >
            <h2 style={titleStyle}>{title}</h2>

            <p style={messageStyle}>{message}</p>

            <div style={buttonRow}>
              <button
                onClick={onCancel}
                disabled={isLoading}
                style={{
                  ...cancelButton,
                  cursor: isLoading ? "not-allowed" : "pointer",
                  opacity: isLoading ? 0.7 : 1,
                }}
              >
                {cancelText}
              </button>

              <button
                onClick={onConfirm}
                disabled={isLoading}
                style={{
                  ...confirmButton,
                  cursor: isLoading ? "not-allowed" : "pointer",
                  opacity: isLoading ? 0.7 : 1,
                }}
              >
                {isLoading ? "Wird ausgeführt..." : confirmText}
              </button>
            </div>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}

export default ConfirmModal;

/* Styles */

const overlayStyle: React.CSSProperties = {
  position: "fixed",
  top: 0,
  left: 0,
  width: "100vw",
  height: "100vh",
  background: "rgba(0, 0, 0, 0.75)",
  backdropFilter: "blur(6px)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 1000,
};

const modalStyle: React.CSSProperties = {
  width: "100%",
  maxWidth: "420px",
  background: theme.colors.panel,
  borderRadius: "20px",
  padding: "26px 24px 22px",
  border: "1px solid rgba(148, 163, 184, 0.18)",
  boxShadow: "0 25px 60px rgba(0, 0, 0, 0.6)",
};

const titleStyle: React.CSSProperties = {
  margin: "0 0 12px 0",
  fontSize: "1.4rem",
  fontWeight: 800,
  color: theme.colors.textPrimary,
};

const messageStyle: React.CSSProperties = {
  margin: "0 0 20px 0",
  fontSize: "0.98rem",
  lineHeight: 1.7,
  color: theme.colors.textSecondary,
};

const buttonRow: React.CSSProperties = {
  display: "flex",
  justifyContent: "flex-end",
  gap: "10px",
};

const cancelButton: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: "999px",
  background: "transparent",
  color: theme.colors.textSecondary,
  border: `1px solid ${theme.glass.subtle.border}`,
  fontWeight: 700,
};

const confirmButton: React.CSSProperties = {
  padding: "10px 14px",
  borderRadius: "999px",
  background: "linear-gradient(135deg, #b91c1c, #dc2626)",
  color: "#ffffff",
  border: "none",
  fontWeight: 800,
  boxShadow: "0 10px 24px rgba(220, 38, 38, 0.25)",
};