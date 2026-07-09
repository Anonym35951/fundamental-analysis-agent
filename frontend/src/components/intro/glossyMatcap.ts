import { CanvasTexture } from "three";

/** Builds a "glossy black studio" matcap procedurally — no external image
 * asset, no CDN fetch. A matcap encodes lighting response purely by the
 * on-screen position of the surface normal. Kept deliberately subtle: a
 * soft, low-contrast highlight rather than a hard scratch/glint, so curved
 * silhouettes pick up a gentle light streak (like the reference image)
 * instead of a harsh white ring. */
export function createGlossyBlackMatcap(): CanvasTexture {
  const size = 256;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d")!;

  ctx.fillStyle = "#050506";
  ctx.fillRect(0, 0, size, size);

  // Sphere shading base: darker toward the silhouette edge, slightly
  // lighter near the center where the matcap sphere faces the camera.
  const base = ctx.createRadialGradient(
    size * 0.5,
    size * 0.5,
    size * 0.05,
    size * 0.5,
    size * 0.5,
    size * 0.5
  );
  base.addColorStop(0, "#1a1c1f");
  base.addColorStop(0.55, "#0a0b0d");
  base.addColorStop(1, "#020203");
  ctx.fillStyle = base;
  ctx.beginPath();
  ctx.arc(size * 0.5, size * 0.5, size * 0.5, 0, Math.PI * 2);
  ctx.fill();

  // Single soft specular highlight, offset toward upper-left — small and
  // low-contrast so it stays a narrow, gentle streak rather than washing out
  // most of the surface. The vast majority of the matcap (and therefore the
  // band) should stay dark.
  const hi1 = ctx.createRadialGradient(
    size * 0.36,
    size * 0.32,
    0,
    size * 0.36,
    size * 0.32,
    size * 0.14
  );
  hi1.addColorStop(0, "rgba(200,206,212,0.32)");
  hi1.addColorStop(0.6, "rgba(170,176,182,0.12)");
  hi1.addColorStop(1, "rgba(170,176,182,0)");
  ctx.fillStyle = hi1;
  ctx.beginPath();
  ctx.arc(size * 0.5, size * 0.5, size * 0.5, 0, Math.PI * 2);
  ctx.fill();

  // Faint secondary fill light on the opposite side, barely there — just
  // enough to keep the far side from reading as pure flat black.
  const hi2 = ctx.createRadialGradient(
    size * 0.72,
    size * 0.76,
    0,
    size * 0.72,
    size * 0.76,
    size * 0.16
  );
  hi2.addColorStop(0, "rgba(110,118,125,0.1)");
  hi2.addColorStop(1, "rgba(110,118,125,0)");
  ctx.fillStyle = hi2;
  ctx.beginPath();
  ctx.arc(size * 0.5, size * 0.5, size * 0.5, 0, Math.PI * 2);
  ctx.fill();

  // Very subtle brightening right at the outer (grazing-angle) zone — maps
  // to the band's edges/silhouette — thin and low-alpha so it traces the
  // curve as a faint light line, not a bright rim.
  const R = size * 0.5;
  const rim = ctx.createRadialGradient(R, R, R * 0.88, R, R, R);
  rim.addColorStop(0, "rgba(255,255,255,0)");
  rim.addColorStop(0.75, "rgba(200,206,212,0.08)");
  rim.addColorStop(1, "rgba(205,211,217,0.22)");
  ctx.fillStyle = rim;
  ctx.beginPath();
  ctx.arc(R, R, R, 0, Math.PI * 2);
  ctx.fill();

  const texture = new CanvasTexture(canvas);
  texture.needsUpdate = true;
  return texture;
}
