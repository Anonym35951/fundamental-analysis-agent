import React from "react";

export function highlight(text: string, query: string) {
  if (!query) return text;

  const q = query.trim();
  if (!q) return text;

  const regex = new RegExp(`(${q})`, "ig");
  const parts = text.split(regex);

  return (
    <>
      {parts.map((part, i) =>
        regex.test(part) ? (
          <mark
            key={i}
            style={{
              background: "rgba(255,255,255,0.22)",
              color: "white",
              padding: "0 4px",
              borderRadius: 6,
              backdropFilter: "blur(6px)",
            }}
          >
            {part}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}