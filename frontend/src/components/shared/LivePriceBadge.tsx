import { motion } from "framer-motion";
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
      <motion.span
        style={{ ...dot, ...sizing.dot }}
        animate={{ opacity: [1, 0.35, 1] }}
        transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
      />
      {isLoading && price === null ? (
        <span style={{ ...placeholderText, ...sizing.text }}>…</span>
      ) : price !== null ? (
        <span style={{ ...priceText, ...sizing.text }}>${price.toFixed(2)}</span>
      ) : null}
    </span>
  );
}

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
  fontWeight: 800,
  color: theme.colors.textPrimary,
  fontVariantNumeric: "tabular-nums" as const,
};

const placeholderText = {
  color: theme.colors.textMuted,
};

const smSizing = {
  wrapper: {},
  dot: { width: "6px", height: "6px" },
  text: { fontSize: "0.82rem" },
};

const mdSizing = {
  wrapper: {},
  dot: { width: "8px", height: "8px" },
  text: { fontSize: "1.1rem" },
};
