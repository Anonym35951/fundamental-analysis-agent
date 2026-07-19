import { useEffect, type ReactNode } from "react";
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

/** Generic backdrop-fade + scale-in modal shell, reused by every dialog in
 * the app (history list, confirm dialogs, future modals) instead of each
 * one re-implementing entrance animation and glass surface separately. */
export default function Modal({ isOpen, onClose, title, children, maxWidth = "560px" }: ModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    // Verhindert Scroll-Bleed des Hintergrunds hinter dem Overlay (v.a. auf
    // iOS Safari, wo der Body sonst hinter dem fixed-Overlay weiterscrollt).
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [isOpen]);

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
  // inset:0 statt top/left + 100vw/100vh: bleibt auf iOS Safari am
  // sichtbaren (visuellen) Viewport verankert, unabhängig von der
  // Toolbar-Dynamik, die 100vh/100vw gegen den GROSSEN Viewport berechnen
  // würde (siehe RESPONSIVE.md R-P0-2).
  inset: 0,
  background: "rgba(0, 0, 0, 0.78)",
  backdropFilter: "blur(6px)",
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 1000,
  padding: "20px",
  paddingBottom: "calc(20px + env(safe-area-inset-bottom))",
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
  // dvh statt vh: auf iOS Safari folgt die Höhe der tatsächlich sichtbaren
  // Fläche (Toolbar ein-/ausgeblendet), statt gegen den großen Viewport zu
  // rechnen und den unteren Bestätigungs-Button hinter die Toolbar zu
  // schieben. Der Body scrollt intern (siehe bodyStyle), falls der Inhalt
  // trotzdem nicht passt.
  maxHeight: "min(80vh, calc(100dvh - 40px - env(safe-area-inset-bottom)))",
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
