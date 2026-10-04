/**
 * Intensity-weighted centroid in pixel coordinates. A blank image returns
 * the canvas center so downstream centering is a no-op.
 * Mirrors src-tauri/src/nn/preprocess.rs line for line (parity < 1e-9).
 */
export function centerOfMass(img, w, h) {
  let sum = 0;
  let sx = 0;
  let sy = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = img[y * w + x];
      sum += v;
      sx += x * v;
      sy += y * v;
    }
  }
  if (sum === 0) return [(w - 1) / 2, (h - 1) / 2];
  return [sx / sum, sy / sum];
}

/** Bilinearly resample to 13x13 with the given shift, normalized to [0, 1]. */
export function resampleTo13(img, w, h, dx, dy) {
  const OUT = 13;
  const [cx, cy] = centerOfMass(img, w, h);
  const out = new Array(OUT * OUT).fill(0);
  const scaleX = w / OUT;
  const scaleY = h / OUT;
  for (let oy = 0; oy < OUT; oy++) {
    for (let ox = 0; ox < OUT; ox++) {
      const sx = (ox + 0.5) * scaleX - dx - 0.5;
      const sy = (oy + 0.5) * scaleY - dy - 0.5;
      const v = bilinear(img, w, h, sx, sy);
      out[oy * OUT + ox] = v / 255;
    }
  }
  return out;
}

/** Translate so the centroid lands on canvas center, then resample. */
export function preprocess(img, w, h) {
  const [cx, cy] = centerOfMass(img, w, h);
  const dx = (w - 1) / 2 - cx;
  const dy = (h - 1) / 2 - cy;
  return resampleTo13(img, w, h, dx, dy);
}

/** Bilinear sample with clamped edges; outside the canvas reads as zero. */
function bilinear(img, w, h, x, y) {
  if (x < -0.5 || y < -0.5 || x > w - 0.5 || y > h - 0.5) {
    return 0;
  }
  const x0f = Math.floor(x);
  const y0f = Math.floor(y);
  const fx = x - x0f;
  const fy = y - y0f;
  const x0 = x0f;
  const y0 = y0f;
  const x1 = x0 + 1;
  const y1 = y0 + 1;
  const at = (xx, yy) => {
    if (xx < 0 || yy < 0 || xx >= w || yy >= h) return 0;
    return img[yy * w + xx];
  };
  const top = at(x0, y0) * (1 - fx) + at(x1, y0) * fx;
  const bot = at(x0, y1) * (1 - fx) + at(x1, y1) * fx;
  return top * (1 - fy) + bot * fy;
}
