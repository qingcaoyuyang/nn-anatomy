#!/usr/bin/env node
/**
 * Bundles a classroom-ready dataset from the MNIST idx files: 100 train
 * samples and 30 test samples per digit, center-of-mass aligned on the
 * original 28x28 image, downsampled to 13x13, values quantized to 2 decimals
 * to keep the embedded JSON compact.
 *
 * Run: node scripts/generate-dataset.js /path/to/mnist-idx-dir
 * Output: src-tauri/resources/sample-dataset.json
 */
import fs from "node:fs";
import path from "node:path";

const idxDir = process.argv[2] || "/tmp/nn-demo-exp";
const outPath = path.resolve(process.argv[3] ?? "src-tauri/resources/sample-dataset.json");

function readIdx(file) {
  const b = fs.readFileSync(file);
  const ndim = b.readUInt32BE(0) & 0xff;
  const dims = [];
  for (let i = 0; i < ndim; i++) dims.push(b.readUInt32BE(4 * (i + 1)));
  return { dims, data: b.subarray(4 * (ndim + 1)) };
}

function img28(set, i) {
  const off = i * 784;
  const a = new Float64Array(784);
  for (let p = 0; p < 784; p++) a[p] = set.data[off + p] / 255;
  return a;
}

// Center by intensity-weighted mass on 28x28 (threshold 0.05 to skip noise),
// then average 2x2 blocks down to 13x13.
function centered13(im) {
  let mx = 0, my = 0, m = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = im[y * 28 + x];
      if (v > 0.05) { mx += x * v; my += y * v; m += v; }
    }
  }
  if (m === 0) return down13(im);
  mx /= m; my /= m;
  const tx = 13.5 - mx, ty = 13.5 - my;
  const shifted = new Float64Array(784);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const sx = x - tx, sy = y - ty;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const fx = sx - x0, fy = sy - y0;
      const at = (xx, yy) => (xx >= 0 && xx < 28 && yy >= 0 && yy < 28) ? im[yy * 28 + xx] : 0;
      shifted[y * 28 + x] =
        at(x0, y0) * (1 - fx) * (1 - fy) + at(x0 + 1, y0) * fx * (1 - fy) +
        at(x0, y0 + 1) * (1 - fx) * fy + at(x0 + 1, y0 + 1) * fx * fy;
    }
  }
  return down13(shifted);
}

function down13(im) {
  const o = new Float64Array(169);
  let k = 0;
  for (let y = 0; y < 13; y++) {
    for (let x = 0; x < 13; x++) {
      let s = 0;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) s += im[(y * 2 + dy) * 28 + (x * 2 + dx)];
      }
      o[k++] = s / 4;
    }
  }
  return o;
}

// Deterministic per-class sampling: LCG shuffle over the index range, then
// take the first n indices of each class in shuffled order.
function pickPerClass(labels, count, seed) {
  const order = Array.from({ length: labels.length }, (_, i) => i);
  let s = seed;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
  for (let i = order.length - 1; i > 0; i--) {
    const j = (rand() * (i + 1)) | 0;
    [order[i], order[j]] = [order[j], order[i]];
  }
  const byClass = Array.from({ length: 10 }, () => []);
  for (const i of order) {
    const label = labels[i];
    if (byClass[label].length < count) byClass[label].push(i);
  }
  return byClass.flat();
}

const q2 = (v) => Math.round(v * 100) / 100;

const ti = readIdx(path.join(idxDir, "train-images-idx3-ubyte"));
const tl = readIdx(path.join(idxDir, "train-labels-idx1-ubyte"));
const ki = readIdx(path.join(idxDir, "t10k-images-idx3-ubyte"));
const kl = readIdx(path.join(idxDir, "t10k-labels-idx1-ubyte"));

const trainIdx = pickPerClass(tl.data, 100, 42);
const testIdx = pickPerClass(kl.data, 30, 7);

const mk = (images, labels, idxList) => idxList.map((i, n) => ({
  id: `mn-${i}`,
  label: labels[i],
  pixels: Array.from(centered13(img28(images, i)), q2),
  source: "mnist-import",
}));

const dataset = {
  version: 1,
  train: mk(ti, tl.data, trainIdx),
  test: mk(ki, kl.data, testIdx),
};

fs.mkdirSync(path.dirname(outPath), { recursive: true });
fs.writeFileSync(outPath, JSON.stringify(dataset));
console.log(`train=${dataset.train.length} test=${dataset.test.length} -> ${outPath}`);
console.log(`size: ${(fs.statSync(outPath).size / 1024).toFixed(0)} KB`);
