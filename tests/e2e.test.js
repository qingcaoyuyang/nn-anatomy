import test from 'node:test';
import assert from 'node:assert/strict';
import { createNetwork, forward, backward } from '../app/src/nn/network.js';
import { createAdam, adamStep } from '../app/src/nn/optimizer.js';

// E2E in the browser-less sense: the JS teaching engine walks the same
// pipeline the UI drives (preprocess -> forward -> train epochs ->
// evaluate on held-out samples), using the bundled sample dataset.

async function loadDataset() {
  const fs = await import('node:fs/promises');
  const text = await fs.readFile(
    new URL('../src-tauri/resources/sample-dataset.json', import.meta.url),
    'utf8',
  );
  return JSON.parse(text);
}

test('e2e: bundled dataset trains to >90% and round-trips a model', async () => {
  const ds = await loadDataset();
  assert.equal(ds.train.length, 1000);
  assert.equal(ds.test.length, 300);

  const net = createNetwork({ inputs: 169, hidden: 16, outputs: 10 }, 999);
  const adam = createAdam(169 * 16 + 16 + 10 * 16 + 10);

  // 15 epochs at lr=0.001 mirrors one classroom session of bulk training
  for (let epoch = 0; epoch < 15; epoch++) {
    const order = shuffle(1000, epoch);
    for (const idx of order) {
      const s = ds.train[idx];
      const f = forward(net, s.pixels);
      const grads = backward(net, s.pixels, s.label, f);
      adamStep(net, adam, grads, 0.001);
    }
  }

  // evaluate on the held-out test split
  let correct = 0;
  for (const s of ds.test) {
    const f = forward(net, s.pixels);
    const arg = f.p.indexOf(Math.max(...f.p));
    if (arg === s.label) correct += 1;
  }
  const acc = correct / ds.test.length;
  assert.ok(acc > 0.75, `15 epochs accuracy ${(acc * 100).toFixed(1)}% should beat untrained baseline`);

  // 150 epochs reaches the validated >90% classroom bar
  for (let epoch = 15; epoch < 150; epoch++) {
    const order = shuffle(1000, epoch);
    for (const idx of order) {
      const s = ds.train[idx];
      const f = forward(net, s.pixels);
      const grads = backward(net, s.pixels, s.label, f);
      adamStep(net, adam, grads, 0.001);
    }
  }
  let correct2 = 0;
  for (const s of ds.test) {
    const f = forward(net, s.pixels);
    const arg = f.p.indexOf(Math.max(...f.p));
    if (arg === s.label) correct2 += 1;
  }
  assert.ok(correct2 / ds.test.length > 0.9, `150 epochs accuracy ${(correct2 / 300 * 100).toFixed(1)}% must exceed 90%`);

  // model round-trip: serialize, reload, same predictions
  const saved = JSON.stringify({ arch: net.arch, W1: net.W1, b1: net.b1, W2: net.W2, b2: net.b2 });
  const reloaded = JSON.parse(saved);
  const f0 = forward(net, ds.test[0].pixels);
  const f1 = forward(reloaded, ds.test[0].pixels);
  for (let i = 0; i < 10; i++) {
    assert.ok(Math.abs(f0.p[i] - f1.p[i]) < 1e-12, 'round-trip probabilities must match');
  }
});

function shuffle(n, epoch) {
  const order = Array.from({ length: n }, (_, i) => i);
  let s = (epoch * 2654435761 + 42) & 0x7fffffff;
  for (let i = n - 1; i > 0; i--) {
    s = (Math.imul(s, 1103515245) + 12345) & 0x7fffffff;
    const j = Math.floor((s / 0x7fffffff) * (i + 1)) % (i + 1);
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}
