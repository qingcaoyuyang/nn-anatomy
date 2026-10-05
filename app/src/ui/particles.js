/**
 * Particle pool shared by every stage animation (forward inference and
 * backward training). Particles travel along an edge in logical canvas
 * coordinates; the stage calls update(dt)/draw(ctx) once per frame.
 *
 * Rendering uses a two-circle glow trick (cheap outer halo + bright core)
 * instead of shadowBlur, which would tank the frame rate once hundreds of
 * particles share the canvas with the weight graph.
 */

const TAU = Math.PI * 2;

/** Comfortable ceiling for both forward passes plus a reverse flow. */
export const MAX_PARTICLES = 600;

export function createParticleSystem(max = MAX_PARTICLES) {
  const particles = [];

  return {
    count() {
      return particles.length;
    },

    spawn(edge, opts = {}) {
      if (particles.length >= max) particles.shift();
      particles.push({
        x1: edge.x1,
        y1: edge.y1,
        x2: edge.x2,
        y2: edge.y2,
        t: opts.t0 || 0,
        speed: opts.speed || 3,
        size: opts.size || 2.2,
        color: opts.color || '255,160,60',
        alpha: opts.alpha == null ? 0.9 : opts.alpha,
      });
    },

    update(dt) {
      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.t += p.speed * dt;
        if (p.t >= 1) particles.splice(i, 1);
      }
    },

    draw(ctx) {
      for (const p of particles) {
        const x = p.x1 + (p.x2 - p.x1) * p.t;
        const y = p.y1 + (p.y2 - p.y1) * p.t;
        const fade = p.t < 0.85 ? 1 : (1 - p.t) / 0.15;
        const a = p.alpha * fade;
        ctx.beginPath();
        ctx.arc(x, y, p.size * 2.2, 0, TAU);
        ctx.fillStyle = 'rgba(' + p.color + ',' + (a * 0.22).toFixed(3) + ')';
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, p.size, 0, TAU);
        ctx.fillStyle = 'rgba(' + p.color + ',' + a.toFixed(3) + ')';
        ctx.fill();
      }
    },

    clear() {
      particles.length = 0;
    },
  };
}
