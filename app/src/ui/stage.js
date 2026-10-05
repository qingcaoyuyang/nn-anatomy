import { forward } from '../nn/network.js';
import { fireEdge, cancelAll, wait } from './particles.js';

/**
 * The network stage: 256-input pixel wall (16x16), 24 hidden neurons,
 * 10 output classes. Renders static weight lines (top-12 |w| per hidden
 * node, blue=positive red=negative, width=|w|) and plays propagation
 * animations driven by the teaching engine's forward pass.
 */

const POS_COLOR = '#2563EB'; // positive weight
const NEG_COLOR = '#DC2626'; // negative weight
const INK = '#1F2937';
const ORANGE = '#F97316';
const LAYER_MS = 300;

export function initStage(canvas, net, opts = {}) {
  const ctx = canvas.getContext('2d');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let selectedHidden = -1;
  let lastX = new Array(256).fill(0);
  let lastF = null;
  let outputBars = null; // {p: [10], t: progress 0..1}

  const pos = layout();
  function layout() {
    const W = canvas.width;
    const H = canvas.height;
    const inNodes = [];
    for (let i = 0; i < 256; i++) {
      const gx = i % 16;
      const gy = Math.floor(i / 16);
      inNodes.push([W * 0.08 + (gx - 7.5) * 0.0088 * W, H * (0.07 + gy * 0.86 / 15)]);
    }
    const hidNodes = [];
    for (let h = 0; h < 24; h++) hidNodes.push([W * 0.6, H * ((h + 1) / 25)]);
    const outNodes = [];
    for (let o = 0; o < 10; o++) outNodes.push([W * 0.88, H * ((o + 1) / 11)]);
    return { inNodes, hidNodes, outNodes };
  }

  // strongest 12 input connections of hidden node h
  function topEdges(h) {
    const rows = [];
    for (let j = 0; j < 256; j++) rows.push({ w: net.W1[h * 256 + j], j });
    rows.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    return rows.slice(0, 12);
  }

  // strongest 8 hidden->output connections of output node o
  function topEdgesOut(o) {
    const rows = [];
    for (let h = 0; h < 24; h++) rows.push({ w: net.W2[o * 24 + h], h });
    rows.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    return rows.slice(0, 8);
  }

  function drawBase() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    // input pixel wall
    for (let i = 0; i < 256; i++) {
      const [x, y] = pos.inNodes[i];
      const v = lastX[i];
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = v > 0 ? `rgba(249,115,22,${Math.min(0.2 + v * 0.8, 1).toFixed(3)})` : 'rgba(55,65,81,0.12)';
      ctx.fill();
    }
    // input -> hidden edges
    for (let h = 0; h < 24; h++) {
      for (const e of topEdges(h)) {
        const [x1, y1] = pos.inNodes[e.j];
        const [x2, y2] = pos.hidNodes[h];
        ctx.strokeStyle = e.w > 0 ? POS_COLOR : NEG_COLOR;
        ctx.globalAlpha = 0.15;
        ctx.lineWidth = Math.max(Math.min(Math.abs(e.w) * 2, 2.5), 0.4);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
    // hidden -> output edges
    for (let o = 0; o < 10; o++) {
      for (const e of topEdgesOut(o)) {
        const [x1, y1] = pos.hidNodes[e.h];
        const [x2, y2] = pos.outNodes[o];
        ctx.strokeStyle = e.w > 0 ? POS_COLOR : NEG_COLOR;
        ctx.globalAlpha = 0.2;
        ctx.lineWidth = Math.max(Math.min(Math.abs(e.w) * 1.5, 2), 0.4);
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
      }
    }
    ctx.globalAlpha = 1;
    // hidden nodes
    for (let h = 0; h < 24; h++) {
      const [x, y] = pos.hidNodes[h];
      const a = lastF ? lastF.a1[h] : 0;
      const glow = Math.min(Math.abs(a) / 1.5, 1);
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = INK;
      ctx.globalAlpha = 0.25 + glow * 0.75;
      ctx.fill();
      ctx.globalAlpha = 1;
      if (selectedHidden === h) {
        ctx.beginPath();
        ctx.arc(x, y, 10, 0, Math.PI * 2);
        ctx.strokeStyle = ORANGE;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
    // output nodes with labels
    ctx.font = '12px sans-serif';
    for (let o = 0; o < 10; o++) {
      const [x, y] = pos.outNodes[o];
      const p = outputBars ? outputBars.p[o] : (lastF ? lastF.p[o] : 0.1);
      ctx.beginPath();
      ctx.arc(x, y, 5 + p * 8, 0, Math.PI * 2);
      ctx.fillStyle = ORANGE;
      ctx.globalAlpha = 0.25 + p;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.fillStyle = INK;
      ctx.textAlign = 'right';
      ctx.fillText(String(o), x - 14, y + 4);
    }
    if (selectedHidden >= 0) drawHeatmapOverlay(selectedHidden);
  }

  // 16x16 weight heatmap of hidden node h (overlay on the pixel wall zone)
  function drawHeatmapOverlay(h) {
    const baseX = pos.inNodes[0][0] - 3;
    const baseY = pos.inNodes[0][1] - 3;
    const stepX = pos.inNodes[1][0] - pos.inNodes[0][0];
    const stepY = pos.inNodes[16][1] - pos.inNodes[0][1];
    const cellW = Math.max(stepX - 1.5, 2);
    const cellH = Math.max(stepY - 1.5, 2);
    for (let i = 0; i < 256; i++) {
      const w = net.W1[h * 256 + i];
      ctx.fillStyle = w > 0 ? POS_COLOR : NEG_COLOR;
      ctx.globalAlpha = Math.min(Math.abs(w) * 3, 1) * 0.9;
      ctx.fillRect(baseX + (i % 16) * stepX, baseY + Math.floor(i / 16) * stepY, cellW, cellH);
    }
    ctx.globalAlpha = 1;
  }

  const api = {
    drawBase,
    setNetwork(n) {
      // swap in a new weights snapshot (e.g. loaded from backend after training)
      net = n;
      drawBase();
    },
    async playInference(x) {
      cancelAll();
      lastX = x.slice();
      lastF = forward(net, x);
      outputBars = null;
      drawBase();
      if (reduced) return lastF;
      // phase 1: input -> hidden particles
      for (let h = 0; h < 24; h++) {
        for (const e of topEdges(h)) {
          if (lastX[e.j] < 0.15) continue;
          const [x1, y1] = pos.inNodes[e.j];
          const [x2, y2] = pos.hidNodes[h];
          fireEdge(ctx, x1, y1, x2, y2, { value: lastX[e.j] * Math.abs(e.w), duration: LAYER_MS });
        }
      }
      await wait(LAYER_MS + 60);
      drawBase();
      // phase 2: hidden -> output particles
      for (let o = 0; o < 10; o++) {
        for (const e of topEdgesOut(o)) {
          if (Math.abs(lastF.a1[e.h]) < 0.1) continue;
          const [x1, y1] = pos.hidNodes[e.h];
          const [x2, y2] = pos.outNodes[o];
          fireEdge(ctx, x1, y1, x2, y2, { value: lastF.a1[e.h] * Math.abs(e.w), duration: LAYER_MS });
        }
      }
      await wait(LAYER_MS + 60);
      // phase 3: output bars converge from uniform 10% to p
      const start = performance.now();
      await new Promise((resolve) => {
        const step = () => {
          const t = Math.min((performance.now() - start) / LAYER_MS, 1);
          outputBars = { p: lastF.p.map((v) => 0.1 + (v - 0.1) * t), t };
          drawBase();
          if (t < 1) requestAnimationFrame(step);
          else resolve();
        };
        step();
      });
      outputBars = null;
      drawBase();
      return lastF;
    },
    playTrainingStep(before, after) {
      // reverse red error particles from output back to hidden
      cancelAll();
      const target = before.y;
      for (let o = 0; o < 10; o++) {
        const err = (o === target ? 1 : 0) - before.p[o];
        if (Math.abs(err) < 0.05) continue;
        for (const e of topEdgesOut(o)) {
          const [x1, y1] = pos.outNodes[o];
          const [x2, y2] = pos.hidNodes[e.h];
          fireEdge(ctx, x1, y1, x2, y2, { value: Math.abs(err), color: NEG_COLOR, duration: LAYER_MS });
        }
      }
      return wait(LAYER_MS + 60);
    },
    highlightNode(layer, idx) {
      selectedHidden = layer === 1 ? idx : -1;
      drawBase();
    },
  };
  drawBase();
  return api;
}
