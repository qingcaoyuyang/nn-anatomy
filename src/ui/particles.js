/**
 * Lightweight particle system for the network stage: dots that travel
 * along an edge from (x1,y1) to (x2,y2), with optional per-particle
 * alpha. One requestAnimationFrame loop drives all active particles.
 */

const active = new Set();
let rafId = null;

function ensureLoop() {
  if (rafId != null) return;
  const step = () => {
    const now = performance.now();
    for (const fx of active) {
      fx.tick(now);
      if (fx.done) active.delete(fx);
    }
    if (active.size === 0) {
      rafId = null;
      return;
    }
    rafId = requestAnimationFrame(step);
  };
  rafId = requestAnimationFrame(step);
}

function run(duration, eased, onDone) {
  ensureLoop();
  const fx = {
    done: false,
    tick(now) {
      if (fx.t0 == null) fx.t0 = now;
      const t = Math.min((now - fx.t0) / duration, 1);
      eased(t, t === 1);
      if (t === 1) {
        fx.done = true;
        if (onDone) onDone();
      }
    },
  };
  active.add(fx);
  return fx;
}

export function cancelAll() {
  for (const fx of active) fx.done = true;
  active.clear();
  if (rafId != null) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
}

/**
 * Fire one particle along an edge. value scales the dot size and the
 * traversal speed so strong edges feel "heavier" traffic.
 */
export function fireEdge(ctx, x1, y1, x2, y2, opts = {}) {
  const { value = 1, color = '#F97316', duration = 300 } = opts;
  const mag = Math.abs(value);
  const r = 1.5 + Math.min(mag, 1.5) * 1.8;
  const alpha = Math.min(0.25 + mag * 0.75, 1);
  return run(duration, (t) => {
    const x = x1 + (x2 - x1) * t;
    const y = y1 + (y2 - y1) * t;
    // fade out near arrival so particles sink into the node
    const a = alpha * (t < 0.85 ? 1 : (1 - t) / 0.15);
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = color;
    ctx.globalAlpha = Math.max(a, 0);
    ctx.fill();
    ctx.globalAlpha = 1;
  });
}

/**
 * timed wait helper for sequential layer animations.
 */
export function wait(ms) {
  return new Promise((resolve) => {
    run(ms, () => {}, resolve);
  });
}
