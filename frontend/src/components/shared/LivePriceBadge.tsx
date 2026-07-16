import { theme } from "../ui/theme";
import { useLivePrice } from "../../hooks/useLivePrice";

type Props = {
  symbol: string;
  size?: "sm" | "md";
};

/** Small live-price indicator, polled via useLivePrice. Fails silently (no
 * error text) since it's a secondary detail shown next to a symbol in many
 * places (sidebar favorites, dashboard history, analyze results) — a price
 * lookup hiccup shouldn't visually compete with the surrounding content. */
export default function LivePriceBadge({ symbol, size = "sm" }: Props) {
  const { price, error, isLoading } = useLivePrice(symbol);

  if (error) {
    return null;
  }

  const sizing = size === "md" ? mdSizing : smSizing;

  return (
    <span style={{ ...wrapper, ...sizing.wrapper }}>
      <span style={{ ...dot, ...sizing.dot, ...pulseAnimation }} />
      {isLoading && price === null ? (
        <span style={{ ...placeholderText, ...sizing.text }}>…</span>
      ) : price !== null ? (
        <span style={{ ...priceText, ...sizing.text }}>${price.toFixed(2)}</span>
      ) : null}
    </span>
  );
}

// EV-111: reine CSS-Keyframe-Animation (index.css) statt eines dauerhaften
// framer-motion-Ticker pro Badge-Instanz — läuft auf dem Compositor statt
// per JS-Loop und respektiert automatisch prefers-reduced-motion (global in
// index.css geregelt, framer-motion tat das hier nicht).
const pulseAnimation = {
  animationName: "live-price-pulse",
  animationDuration: "1.8s",
  animationTimingFunction: "ease-in-out",
  animationIterationCount: "infinite",
};

const wrapper = {
  display: "inline-flex",
  alignItems: "center",
  gap: "6px",
};

const dot = {
  display: "inline-block",
  borderRadius: theme.radius.pill,
  background: theme.colors.success,
};

const priceText = {
  display: "inline-block",
  fontWeight: 800,
  color: theme.colors.textPrimary,
  fontVariantNumeric: "tabular-nums" as const,
};

const placeholderText = {
  display: "inline-block",
  color: theme.colors.textMuted,
};

// EV-111: feste Mindestbreite, damit der Preis beim Eintreffen (Wechsel von
// "…" auf z. B. "$123.45") keinen Layout-Shift verursacht — "7ch" deckt auch
// vierstellige Kurse mit Cent-Betrag ("$1234.56") ab.
const smSizing = {
  wrapper: {},
  dot: { width: "6px", height: "6px" },
  text: { fontSize: "0.82rem", minWidth: "7ch" },
};

const mdSizing = {
  wrapper: {},
  dot: { width: "8px", height: "8px" },
  text: { fontSize: "1.1rem", minWidth: "7ch" },
};
