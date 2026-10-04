import { test } from "node:test";
import assert from "node:assert/strict";
import { createNetwork, forward, backward } from "../src/nn/network.js";

test("createNetwork produces deterministic weights for a seed", () => {
  const a = createNetwork({ inputs: 169, hidden: 16, outputs: 10 }, 42);
  const b = createNetwork({ inputs: 169, hidden: 16, outputs: 10 }, 42);
  assert.deepEqual(a.W1, b.W1);
  assert.deepEqual(a.b1, b.b1);
  assert.deepEqual(a.W2, b.W2);
  assert.deepEqual(a.b2, b.b2);
  const c = createNetwork({ inputs: 169, hidden: 16, outputs: 10 }, 43);
  assert.notDeepEqual(a.W1, c.W1);
});

test("forward outputs a valid softmax distribution", () => {
  const net = createNetwork({ inputs: 169, hidden: 16, outputs: 10 }, 42);
  const x = new Array(169).fill(0.5);
  const f = forward(net, x);
  assert.equal(f.p.length, 10);
  assert.equal(f.z1.length, 16);
  assert.equal(f.a1.length, 16);
  assert.equal(f.z2.length, 10);
  const sum = f.p.reduce((acc, v) => acc + v, 0);
  assert.ok(Math.abs(sum - 1) < 1e-9);
  assert.ok(f.p.every((v) => v > 0 && v < 1));
});

test("backward gradients match numerical differences", () => {
  const net = createNetwork({ inputs: 169, hidden: 16, outputs: 10 }, 7);
  const x = Array.from({ length: 169 }, (_, i) => ((i * 37) % 13) / 13);
  const y = 3;
  const f = forward(net, x);
  const g = backward(net, x, y, f);
  const eps = 1e-5;
  // Central difference on L = -ln p[y] for one weight in each tensor.
  const cases = [
    ["W1", 3 * 169 + 10, g.gW1[3 * 169 + 10]],
    ["W2", 4, g.gW2[4]],
    ["b1", 5, g.gb1[5]],
  ];
  for (const [field, idx, ana] of cases) {
    const up = structuredClone(net);
    const dn = structuredClone(net);
    up[field][idx] += eps;
    dn[field][idx] -= eps;
    const lp = -Math.log(forward(up, x).p[y]);
    const ld = -Math.log(forward(dn, x).p[y]);
    const num = (lp - ld) / (2 * eps);
    assert.ok(Math.abs(num - ana) < 1e-6, `${field}[${idx}]: ${num} vs ${ana}`);
  }
});
