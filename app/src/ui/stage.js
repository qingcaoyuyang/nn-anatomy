import { forward } from '../nn/network.js';
import { createParticleSystem, MAX_PARTICLES } from './particles.js';

/**
 * Network stage: 256->24->10 MLP rendered on a 900x520 logical canvas.
 *
 * Layout: 16x16 input pixel wall (left), 24 hidden neurons (middle),
 * 10 output probability bars (right). Edges: top-12 |w1| per hidden node,
 * all 24x10 output edges. Blue = positive weight, red = negative, width
 * proportional to |w|.
 *
 * Inference animation: particles flow input->hidden->output in two waves,
 * hidden nodes glow up with their activation, output bars converge from a
 * uniform 10% to the softmax result.
 *
 * Training animation: red error particles flow backwards (output->hidden
 * ->input), the 24 edges with the largest |dw| emit expanding pulse rings,
 * and the output bars show a before (grey) / after (blue) comparison.
 *
 * prefers-reduced-motion collapses every animation to its final frame.
 * Re-invoking playInference/playTrainingStep while one is running returns
 * the in-flight promise instead of stacking a second animation.
 */

const TAU = Math.PI * 2;

const W = 900;
const H = 520;

const WALL = { x: 36, y: 130, cell: 17, n: 16 };
const HIDDEN = { x: 470, y0: 38, y1: 482, r: 11, n: 24 };
const OUT = { labelX: 640, barX: 672, barMax: 130, y0: 60, rowH: 42, n: 10 };

const POS = '37,99,235';
const NEG = '220,38,38';
const INK = '#1c1917';
const MUTED = '#78716c';
const GRID_BG = '#fafaf9';
const ORANGE = '249,115,22';

const FONT_NUM = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
const FONT_NUM_SM = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
const FONT_LABEL = '12px "Noto Sans SC", sans-serif';

const clamp01 = (v) => (v < 0 ? 0 : v > 1 ? 1 : v);
const smooth = (t) => {
  const c = clamp01(t);
  return c * c * (3 - 2 * c);
};

function hiddenY(h) {
  return HIDDEN.y0 + (h * (HIDDEN.y1 - HIDDEN.y0)) / (HIDDEN.n - 1);
}

function outY(o) {
  return OUT.y0 + o * OUT.rowH + OUT.rowH / 2;
}

function pixelXY(i) {
  const c = i % WALL.n;
  const r = (i / WALL.n) | 0;
  return [WALL.x + c * WALL.cell + WALL.cell / 2, WALL.y + r * WALL.cell + WALL.cell / 2];
}

function topKAbs(arr, off, n, k) {
  const idx = new Array(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  idx.sort((a, b) => Math.abs(arr[off + b]) - Math.abs(arr[off + a]));
  return idx.slice(0, Math.min(k, n));
}

function buildEdges(net) {
  const nIn = net.arch.inputs;
  const nH = net.arch.hidden;
  const nOut = net.arch.outputs;
  const e1 = [];
  let max1 = 1e-9;
  for (let h = 0; h < nH; h++) {
    for (const i of topKAbs(net.W1, h * nIn, nIn, 12)) {
      const w = net.W1[h * nIn + i];
      const aw = Math.abs(w);
      if (aw > max1) max1 = aw;
      const [px, py] = pixelXY(i);
      e1.push({ h, i, x1: px, y1: py, x2: HIDDEN.x, y2: hiddenY(h), w });
    }
  }
  const e2 = [];
  const e2ByOutput = Array.from({ length: nOut }, () => []);
  let max2 = 1e-9;
  for (let o = 0; o < nOut; o++) {
    for (let h = 0; h < nH; h++) {
      const w = net.W2[o * nH + h];
      const aw = Math.abs(w);
      if (aw > max2) max2 = aw;
      const e = { o, h, x1: HIDDEN.x, y1: hiddenY(h), x2: OUT.barX, y2: outY(o), w };
      e2.push(e);
      e2ByOutput[o].push(e);
    }
  }
  return { e1, e2, e2ByOutput, max1, max2 };
}

function fitCanvas(canvas, ctx) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(W * dpr);
  canvas.height = Math.round(H * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

export function initStage(canvas, net) {
  const ctx = canvas.getContext('2d');
  fitCanvas(canvas, ctx);
  const sys = createParticleSystem(MAX_PARTICLES);

  const state = {
    net,
    edges: buildEdges(net),
    x: null,
    trace: null,
    lastTrace: null,
    heatmap: null,
    compare: null,
    pulses: [],
    glowT: 1,
    barT: 1,
    now: null,
    anim: null,
    reduced: window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  };

  function drawEdges() {
    const { e1, e2, max1, max2 } = state.edges;
    for (const e of e1) {
      const k = Math.abs(e.w) / max1;
      ctx.strokeStyle = 'rgba(' + (e.w >= 0 ? POS : NEG) + ',' + (0.1 + 0.28 * k).toFixed(3) + ')';
      ctx.lineWidth = 0.6 + 2.4 * k;
      ctx.beginPath();
      ctx.moveTo(e.x1, e.y1);
      ctx.lineTo(e.x2, e.y2);
      ctx.stroke();
    }
    for (const e of e2) {
      const k = Math.abs(e.w) / max2;
      ctx.strokeStyle = 'rgba(' + (e.w >= 0 ? POS : NEG) + ',' + (0.12 + 0.3 * k).toFixed(3) + ')';
      ctx.lineWidth = 0.6 + 2.4 * k;
      ctx.beginPath();
      ctx.moveTo(e.x1, e.y1);
      ctx.lineTo(e.x2, e.y2);
      ctx.stroke();
    }
  }

  function drawInputWall() {
    ctx.fillStyle = GRID_BG;
    ctx.fillRect(WALL.x, WALL.y, WALL.n * WALL.cell, WALL.n * WALL.cell);
    const nIn = state.net.arch.inputs;
    if (state.heatmap != null) {
      const h = state.heatmap;
      let m = 1e-9;
      for (let i = 0; i < nIn; i++) {
        const a = Math.abs(state.net.W1[h * nIn + i]);
        if (a > m) m = a;
      }
      for (let i = 0; i < nIn; i++) {
        const w = state.net.W1[h * nIn + i];
        const k = clamp01(Math.abs(w) / m);
        const c = Math.round(245 - 200 * k);
        ctx.fillStyle = w >= 0 ? 'rgb(' + c + ',' + c + ',245)' : 'rgb(245,' + c + ',' + c + ')';
        const [px, py] = pixelXY(i);
        ctx.fillRect(px - WALL.cell / 2, py - WALL.cell / 2, WALL.cell, WALL.cell);
      }
    } else if (state.x) {
      for (let i = 0; i < nIn; i++) {
        const g = Math.round(255 * (1 - clamp01(state.x[i])));
        ctx.fillStyle = 'rgb(' + g + ',' + g + ',' + g + ')';
        const [px, py] = pixelXY(i);
        ctx.fillRect(px - WALL.cell / 2, py - WALL.cell / 2, WALL.cell, WALL.cell);
      }
    }
    ctx.strokeStyle = '#e7e5e4';
    ctx.lineWidth = 1;
    ctx.strokeRect(WALL.x, WALL.y, WALL.n * WALL.cell + 1, WALL.n * WALL.cell + 1);
  }

  function drawHidden() {
    const a1 = state.trace ? state.trace.a1 : null;
    let scale = 1e-9;
    if (a1) {
      for (let h = 0; h < HIDDEN.n; h++) {
        const a = Math.abs(a1[h]);
        if (a > scale) scale = a;
      }
    }
    for (let h = 0; h < HIDDEN.n; h++) {
      const cx = HIDDEN.x;
      const cy = hiddenY(h);
      const b = a1 ? clamp01(a1[h] / scale) * state.glowT : 0;
      if (b > 0.01) {
        const g = ctx.createRadialGradient(cx, cy, 1, cx, cy, HIDDEN.r * 2.4);
        g.addColorStop(0, 'rgba(251,146,60,' + (0.85 * b).toFixed(3) + ')');
        g.addColorStop(1, 'rgba(251,146,60,0)');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(cx, cy, HIDDEN.r * 2.4, 0, TAU);
        ctx.fill();
      }
      const shade = Math.round(28 + 200 * (1 - b));
      ctx.beginPath();
      ctx.arc(cx, cy, HIDDEN.r, 0, TAU);
      ctx.fillStyle = 'rgb(' + shade + ',' + Math.round(shade * 0.75) + ',' + Math.round(shade * 0.45) + ')';
      ctx.fill();
      ctx.strokeStyle = '#44403c';
      ctx.lineWidth = 1;
      ctx.stroke();
      if (state.heatmap === h) {
        ctx.beginPath();
        ctx.arc(cx, cy, HIDDEN.r + 3.5, 0, TAU);
        ctx.strokeStyle = '#f59e0b';
        ctx.lineWidth = 2.5;
        ctx.stroke();
      }
    }
  }

  function drawOutput() {
    const tr = state.trace;
    ctx.textBaseline = 'middle';
    for (let o = 0; o < OUT.n; o++) {
      const cy = outY(o);
      ctx.fillStyle = '#57534e';
      ctx.font = FONT_NUM;
      ctx.textAlign = 'right';
      ctx.fillText(String(o), OUT.labelX, cy);
      if (state.compare) {
        const pb = state.compare.before[o];
        const pa = state.compare.after[o];
        ctx.fillStyle = '#a8a29e';
        ctx.fillRect(OUT.barX, cy - 6, pb * OUT.barMax, 5.5);
        ctx.fillStyle = 'rgb(' + POS + ')';
        ctx.fillRect(OUT.barX, cy + 0.5, pa * OUT.barMax, 5.5);
        ctx.fillStyle = '#57534e';
        ctx.font = FONT_NUM_SM;
        ctx.textAlign = 'left';
        ctx.fillText(pa.toFixed(3), OUT.barX + OUT.barMax + 6, cy);
      } else if (tr) {
        const pShow = 0.1 + (tr.p[o] - 0.1) * smooth(state.barT);
        const isTop = o === tr.pred;
        ctx.fillStyle = isTop ? '#f59e0b' : '#64748b';
        ctx.fillRect(OUT.barX, cy - 6, pShow * OUT.barMax, 12);
        ctx.fillStyle = isTop ? INK : '#57534e';
        ctx.font = FONT_NUM_SM;
        ctx.textAlign = 'left';
        ctx.fillText(pShow.toFixed(3), OUT.barX + OUT.barMax + 6, cy);
      } else {
        ctx.fillStyle = '#e7e5e4';
        ctx.fillRect(OUT.barX, cy - 6, 0.1 * OUT.barMax, 12);
      }
    }
  }

  function drawPulses() {
    if (state.now == null) return;
    for (const pl of state.pulses) {
      const pt = (state.now - pl.t0) / pl.dur;
      if (pt < 0 || pt > 1) continue;
      const r = 3 + 17 * pt;
      const a = 0.75 * (1 - pt);
      ctx.beginPath();
      ctx.arc(pl.x, pl.y, r, 0, TAU);
      ctx.strokeStyle = 'rgba(' + (pl.dw >= 0 ? POS : NEG) + ',' + a.toFixed(3) + ')';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  function drawCaptions() {
    ctx.fillStyle = MUTED;
    ctx.font = FONT_LABEL;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(state.heatmap != null ? '输入墙：h' + state.heatmap + ' 权重热力图（点节点切换回输入）' : '输入 · 16x16', WALL.x, WALL.y - 12);
    ctx.textAlign = 'center';
    ctx.fillText('隐藏层 · 24', HIDDEN.x, HIDDEN.y0 - 18);
    ctx.textAlign = 'left';
    ctx.fillText('输出 · 概率', OUT.barX, OUT.y0 - 14);
    ctx.textAlign = 'left';
    if (state.compare) {
      ctx.fillStyle = '#a8a29e';
      ctx.fillRect(W - 210, H - 26, 10, 10);
      ctx.fillStyle = 'rgb(' + POS + ')';
      ctx.fillRect(W - 118, H - 26, 10, 10);
      ctx.fillStyle = MUTED;
      ctx.fillText('更新前', W - 196, H - 17);
      ctx.fillText('更新后', W - 104, H - 17);
    } else if (state.trace) {
      ctx.fillStyle = MUTED;
      ctx.fillText('预测 ' + state.trace.pred + '  p=' + state.trace.p[state.trace.pred].toFixed(3), WALL.x, H - 17);
    } else {
      ctx.fillStyle = MUTED;
      ctx.fillText('等待推理…', WALL.x, H - 17);
    }
  }

  function render() {
    ctx.clearRect(0, 0, W, H);
    drawEdges();
    drawPulses();
    drawInputWall();
    drawHidden();
    drawOutput();
    sys.draw(ctx);
    drawCaptions();
  }

  function startAnimation(durationMs, onFrame) {
    if (state.anim) return state.anim.promise;
    if (state.reduced) {
      state.now = durationMs;
      onFrame(1, 0);
      sys.clear();
      state.now = null;
      state.pulses.length = 0;
      state.glowT = 1;
      state.barT = 1;
      render();
      return Promise.resolve();
    }
    let resolvePromise;
    const promise = new Promise((res) => {
      resolvePromise = res;
    });
    const t0 = performance.now();
    let last = t0;
    const anim = { promise, raf: 0, cancelled: false, finish: null, exposed: null };
    state.anim = anim;
    const finish = () => {
      if (state.anim !== anim) return;
      state.anim = null;
      state.now = null;
      state.pulses.length = 0;
      sys.clear();
      state.glowT = 1;
      state.barT = 1;
      render();
      resolvePromise();
    };
    anim.finish = finish;
    const frame = (now) => {
      if (anim.cancelled) return;
      const dt = Math.min(0.1, Math.max(0, (now - last) / 1000));
      last = now;
      const t = Math.min(1, (now - t0) / durationMs);
      state.now = now - t0;
      onFrame(t, dt);
      sys.update(dt);
      render();
      if (t < 1) {
        anim.raf = requestAnimationFrame(frame);
      } else {
        finish();
      }
    };
    anim.raf = requestAnimationFrame(frame);
    return promise;
  }

  const api = {
    setNetwork(next) {
      state.net = next;
      state.edges = buildEdges(next);
      state.x = null;
      state.trace = null;
      state.lastTrace = null;
      state.heatmap = null;
      state.compare = null;
      state.pulses.length = 0;
      sys.clear();
      state.glowT = 1;
      state.barT = 1;
      render();
    },

    render,

    get lastTrace() {
      return state.lastTrace;
    },

    playInference(x) {
      if (state.anim) return state.anim.exposed || state.anim.promise;
      const nIn = state.net.arch.inputs;
      if (!Array.isArray(x) && !(x instanceof Float64Array)) {
        throw new Error('playInference 需要 ' + nIn + ' 维输入数组。');
      }
      if (x.length !== nIn) throw new Error('playInference 需要 ' + nIn + ' 维输入，收到 ' + x.length + '。');
      const trace = forward(state.net, x);
      trace.pred = trace.p.indexOf(Math.max(...trace.p));
      state.x = Array.from(x);
      state.trace = trace;
      state.lastTrace = trace;
      state.compare = null;
      state.heatmap = null;

      const { e1, e2 } = state.edges;
      let maxA1 = 1e-9;
      for (let h = 0; h < HIDDEN.n; h++) {
        const a = Math.abs(trace.a1[h]);
        if (a > maxA1) maxA1 = a;
      }
      const totalA = 260;
      const totalB = 150;
      let spawnedA = 0;
      let spawnedB = 0;

      const spawnFwd1 = () => {
        for (let k = 0; k < 4; k++) {
          const e = e1[(Math.random() * e1.length) | 0];
          const b = clamp01(state.x[e.i]);
          if (Math.random() < 0.12 + 0.88 * b) {
            sys.spawn(e, {
              speed: 2.6 + Math.random(),
              size: 2.2,
              color: '255,' + Math.round(255 - 150 * b) + ',' + Math.round(255 - 200 * b),
              alpha: 0.85,
            });
            return;
          }
        }
      };
      const spawnFwd2 = () => {
        const e = e2[(Math.random() * e2.length) | 0];
        const b = clamp01(trace.a1[e.h] / maxA1);
        sys.spawn(e, {
          speed: 2.6 + Math.random(),
          size: 2.2,
          color: '255,' + Math.round(190 - 90 * b) + ',60',
          alpha: 0.4 + 0.5 * b,
        });
      };

      const exposed = startAnimation(700, (t) => {
        state.glowT = clamp01(t / 0.45);
        state.barT = smooth((t - 0.4) / 0.6);
        if (t < 0.5) {
          const wantA = Math.floor(totalA * (t / 0.5));
          for (; spawnedA < wantA; spawnedA++) spawnFwd1();
        } else {
          for (; spawnedA < totalA; spawnedA++) spawnFwd1();
          const wantB = Math.floor(totalB * ((t - 0.5) / 0.5));
          for (; spawnedB < wantB; spawnedB++) spawnFwd2();
        }
      }).then(() => trace);
      if (state.anim) state.anim.exposed = exposed;
      return exposed;
    },

    playTrainingStep(before, after) {
      if (state.anim) return state.anim.exposed || state.anim.promise;
      if (!before || !before.trace || !after || !after.trace) {
        throw new Error('playTrainingStep 需要 before/after（含 trace、W1、W2 快照）。');
      }
      state.compare = { before: before.trace.p.slice(), after: after.trace.p.slice() };
      state.trace = after.trace;
      state.lastTrace = after.trace;

      const y = before.y != null ? before.y : after.y != null ? after.y : before.trace.pred;
      const err = new Array(OUT.n);
      let errSum = 0;
      for (let o = 0; o < OUT.n; o++) {
        err[o] = Math.abs((o === y ? 1 : 0) - before.trace.p[o]);
        errSum += err[o];
      }
      if (errSum < 1e-9) errSum = 1e-9;

      const nIn = state.net.arch.inputs;
      const nH = state.net.arch.hidden;
      const { e1, e2 } = state.edges;
      const cands = [];
      for (const e of e1) {
        const dw = after.W1[e.h * nIn + e.i] - before.W1[e.h * nIn + e.i];
        cands.push({ x: (e.x1 + e.x2) / 2, y: (e.y1 + e.y2) / 2, dw });
      }
      for (const e of e2) {
        const dw = after.W2[e.o * nH + e.h] - before.W2[e.o * nH + e.h];
        cands.push({ x: (e.x1 + e.x2) / 2, y: (e.y1 + e.y2) / 2, dw });
      }
      cands.sort((a, b) => Math.abs(b.dw) - Math.abs(a.dw));
      state.pulses = cands.slice(0, 24).map((c, k) => ({
        x: c.x,
        y: c.y,
        dw: c.dw,
        t0: 120 + (k / 24) * 480,
        dur: 320,
      }));

      const { e2ByOutput } = state.edges;
      const totalB1 = 260;
      const totalB2 = 140;
      let spawnedB1 = 0;
      let spawnedB2 = 0;

      const spawnBack1 = () => {
        let r = Math.random() * errSum;
        let o = 0;
        while (o < 9 && r > err[o]) {
          r -= err[o];
          o++;
        }
        const edges = e2ByOutput[o];
        const e = edges[(Math.random() * edges.length) | 0];
        const b = clamp01(err[o]);
        sys.spawn({ x1: e.x2, y1: e.y2, x2: e.x1, y2: e.y1 }, {
          speed: 2.4 + Math.random(),
          size: 2.2,
          color: '220,' + Math.round(70 - 40 * b) + ',' + Math.round(70 - 40 * b),
          alpha: 0.5 + 0.4 * b,
        });
      };
      const spawnBack2 = () => {
        const e = e1[(Math.random() * e1.length) | 0];
        sys.spawn({ x1: e.x2, y1: e.y2, x2: e.x1, y2: e.y1 }, {
          speed: 2.4 + Math.random(),
          size: 2,
          color: '220,50,50',
          alpha: 0.6,
        });
      };

      const exposed = startAnimation(950, (t) => {
        state.glowT = 1;
        if (t < 0.5) {
          const want = Math.floor(totalB1 * (t / 0.5));
          for (; spawnedB1 < want; spawnedB1++) spawnBack1();
        } else {
          for (; spawnedB1 < totalB1; spawnedB1++) spawnBack1();
          const want2 = Math.floor(totalB2 * ((t - 0.5) / 0.5));
          for (; spawnedB2 < want2; spawnedB2++) spawnBack2();
        }
      });
      if (state.anim) state.anim.exposed = exposed;
      return exposed;
    },

    highlightNode(layer, idx) {
      if (layer !== 1 && layer !== 'hidden') throw new Error("highlightNode 暂不支持 layer='" + layer + "'。");
      if (!Number.isInteger(idx) || idx < 0 || idx >= HIDDEN.n) {
        throw new Error('hidden 节点下标越界：' + idx + '。');
      }
      state.heatmap = state.heatmap === idx ? null : idx;
      render();
    },
  };

  function hitHidden(lx, ly) {
    for (let h = 0; h < HIDDEN.n; h++) {
      const dx = lx - HIDDEN.x;
      const dy = ly - hiddenY(h);
      if (dx * dx + dy * dy <= (HIDDEN.r + 5) * (HIDDEN.r + 5)) return h;
    }
    return -1;
  }

  function toLogical(ev) {
    const r = canvas.getBoundingClientRect();
    return [(ev.clientX - r.left) * (W / r.width), (ev.clientY - r.top) * (H / r.height)];
  }

  canvas.addEventListener('click', (ev) => {
    const [lx, ly] = toLogical(ev);
    const h = hitHidden(lx, ly);
    if (h >= 0) {
      api.highlightNode(1, h);
    } else if (state.heatmap != null) {
      state.heatmap = null;
      render();
    }
  });

  canvas.addEventListener('mousemove', (ev) => {
    const [lx, ly] = toLogical(ev);
    canvas.style.cursor = hitHidden(lx, ly) >= 0 ? 'pointer' : 'default';
  });

  window.addEventListener('resize', () => {
    fitCanvas(canvas, ctx);
    render();
  });

  render();
  return api;
}
