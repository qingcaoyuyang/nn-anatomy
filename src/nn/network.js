/**
 * Teaching engine: a 1-hidden-layer MLP that mirrors the Rust core in
 * src-tauri/src/nn/network.rs operation for operation so both engines stay
 * within 1e-9 of each other (same seed => same weights => same training run).
 */

/** Canonical mulberry32 (bryc's reference) — matches the Rust implementation. */
function mulberry32(seed) {
  let state = seed >>> 0;
  return function () {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  };
}

/** Folds the full seed into 32 bits so high bits matter (Rust parity). */
function foldSeed(seed) {
  return (Number(seed) % 4294967296 ^ Math.floor(Number(seed) / 4294967296)) >>> 0;
}

/** Box-Muller gaussian, same operation order as Rust. */
function gaussian(rng) {
  const u1 = Math.max((rng() >>> 8) / (1 << 24), 1e-12);
  const u2 = (rng() >>> 8) / (1 << 24);
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

export function createNetwork(arch, seed) {
  const rng = mulberry32(foldSeed(seed));
  const std1 = Math.sqrt(2.0 / arch.inputs);
  const std2 = Math.sqrt(2.0 / arch.hidden);
  const W1 = new Array(arch.hidden * arch.inputs);
  for (let i = 0; i < W1.length; i++) W1[i] = gaussian(rng) * std1;
  const b1 = new Array(arch.hidden).fill(0);
  const W2 = new Array(arch.outputs * arch.hidden);
  for (let i = 0; i < W2.length; i++) W2[i] = gaussian(rng) * std2;
  const b2 = new Array(arch.outputs).fill(0);
  return { arch: { ...arch }, W1, b1, W2, b2 };
}

function leakyRelu(z) {
  return z > 0 ? z : 0.01 * z;
}

function softmax(z) {
  let max = -Infinity;
  for (const v of z) if (v > max) max = v;
  const exps = z.map((v) => Math.exp(v - max));
  let sum = 0;
  for (const e of exps) sum += e;
  return exps.map((e) => e / sum);
}

export function forward(net, x) {
  const { inputs, hidden, outputs } = net.arch;
  // z1 = W1 x + b1 (row-major [hidden x inputs]), a1 = LeakyReLU(0.01)
  const z1 = net.b1.slice();
  for (let h = 0; h < hidden; h++) {
    let acc = 0;
    const row = h * inputs;
    for (let j = 0; j < inputs; j++) acc += net.W1[row + j] * x[j];
    z1[h] += acc;
  }
  const a1 = z1.map(leakyRelu);
  // z2 = W2 a1 + b2 (row-major [outputs x hidden])
  const z2 = net.b2.slice();
  for (let o = 0; o < outputs; o++) {
    let acc = 0;
    const row = o * hidden;
    for (let h = 0; h < hidden; h++) acc += net.W2[row + h] * a1[h];
    z2[o] += acc;
  }
  const p = softmax(z2);
  return { z1, a1, z2, p };
}

export function backward(net, x, y, f) {
  const { inputs, hidden, outputs } = net.arch;
  // dL/dz2 = p - onehot(y)
  const dz2 = new Array(outputs);
  for (let o = 0; o < outputs; o++) dz2[o] = f.p[o] - (o === y ? 1 : 0);
  const gW2 = new Array(outputs * hidden);
  for (let i = 0; i < gW2.length; i++) {
    gW2[i] = dz2[Math.floor(i / hidden)] * f.a1[i % hidden];
  }
  const gb2 = dz2.slice();
  // Backprop through LeakyReLU into the hidden layer.
  const da1 = new Array(hidden);
  for (let h = 0; h < hidden; h++) {
    let acc = 0;
    for (let o = 0; o < outputs; o++) acc += net.W2[o * hidden + h] * dz2[o];
    da1[h] = acc;
  }
  const dz1 = new Array(hidden);
  for (let h = 0; h < hidden; h++) dz1[h] = da1[h] * (f.z1[h] > 0 ? 1 : 0.01);
  const gW1 = new Array(hidden * inputs);
  for (let i = 0; i < gW1.length; i++) {
    gW1[i] = dz1[Math.floor(i / inputs)] * x[i % inputs];
  }
  return { gW1, gb1: dz1, gW2, gb2 };
}
