import type { CSSProperties } from "react";
import { theme } from "../ui/theme";

const sansFont = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

type SectionHeadingProps = {
  eyebrow: string;
  title: string;
  subtitle?: string;
  align?: "left" | "center";
};

/** Reusable eyebrow + title + subtitle block, shared by every marketing
 * section below the hero so heading styling stays consistent without
 * re-declaring the same three inline styles per section. */
export default function SectionHeading({ eyebrow, title, subtitle, align = "center" }: SectionHeadingProps) {
  const isCenter = align === "center";

  return (
    <div style={{ ...wrapper, textAlign: isCenter ? "center" : "left", marginInline: isCenter ? "auto" : 0 }}>
      <div style={eyebrowStyle}>{eyebrow}</div>
      <h2 style={titleStyle}>{title}</h2>
      {subtitle ? (
        <p style={{ ...subtitleStyle, marginInline: isCenter ? "auto" : 0 }}>{subtitle}</p>
      ) : null}
    </div>
  );
}

const wrapper: CSSProperties = {
  marginBottom: "40px",
  maxWidth: "860px",
};

const eyebrowStyle: CSSProperties = {
  display: "inline-block",
  fontSize: "0.78rem",
  fontWeight: 800,
  letterSpacing: "0.08em",
  textTransform: "uppercase",
  color: theme.colors.chrome,
  marginBottom: "14px",
};

const titleStyle: CSSProperties = {
  margin: "0 0 16px 0",
  fontSize: "2.4rem",
  letterSpacing: "-0.04em",
  lineHeight: 1.15,
  fontFamily: sansFont,
  color: theme.colors.textPrimary,
};

const subtitleStyle: CSSProperties = {
  margin: 0,
  color: theme.colors.textSecondary,
  fontSize: "1.14rem",
  lineHeight: 1.9,
  maxWidth: "860px",
};
