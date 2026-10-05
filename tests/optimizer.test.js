import { test } from "node:test";
import assert from "node:assert/strict";
import { createNetwork, forward, backward } from "../app/src/nn/network.js";
import { createAdam, adamStep, sgdStep } from "../app/src/nn/optimizer.js";

test("adam first step moves each parameter by about lr", () => {
  const net = createNetwork({ inputs: 3, hidden: 2, outputs: 2 }, 1);
  const before = structuredClone(net);
  const grads = {
    gW1: new Array(6).fill(0.5),
    gb1: new Array(2).fill(-0.25),
    gW2: new Array(4).fill(1.0),
    gb2: new Array(2).fill(-2.0),
  };
  const state = createAdam(14);
  const lr = 0.1;
  adamStep(net, state, grads, lr);
  assert.equal(state.t, 1);
  // After bias correction the first update is ~sign(g) * lr.
  for (let i = 0; i < net.W1.length; i++) {
    const delta = net.W1[i] - before.W1[i];
    assert.ok(Math.abs(Math.abs(delta) - lr) < 1e-6, `W1[${i}]: ${delta}`);
  }
  assert.ok(Math.abs(state.m[0] - 0.1 * 0.5) < 1e-15);
  assert.ok(Math.abs(state.v[0] - 0.001 * 0.25) < 1e-15);
});

test("sgd step is exactly -lr * grad", () => {
  const net = createNetwork({ inputs: 3, hidden: 2, outputs: 2 }, 1);
  const before = structuredClone(net);
  const grads = {
    gW1: new Array(6).fill(0.25),
    gb1: new Array(2).fill(-0.5),
    gW2: new Array(4).fill(1.5),
    gb2: new Array(2).fill(-2.0),
  };
  const lr = 0.3;
  sgdStep(net, grads, lr);
  for (const field of ["W1", "b1", "W2", "b2"]) {
    for (let i = 0; i < net[field].length; i++) {
      const g = { W1: "gW1", b1: "gb1", W2: "gW2", b2: "gb2" }[field];
      assert.equal(net[field][i], before[field][i] - lr * grads[g][i]);
    }
  }
});

test("100 steps of adam training on 100 samples reduces loss", () => {
  const net = createNetwork({ inputs: 4, hidden: 8, outputs: 3 }, 9);
  // 100 synthetic samples: class k has feature k pushed high.
  const samples = [];
  for (let i = 0; i < 100; i++) {
    const y = i % 3;
    const x = [0.1, -0.1, 0.05, -0.05];
    x[y] = 0.9;
    samples.push({ x, y });
  }
  const loss = (n) => {
    let total = 0;
    for (const s of samples) total += -Math.log(forward(n, s.x).p[s.y]);
    return total / samples.length;
  };
  const initial = loss(net);
  const state = createAdam(4 * 8 + 8 + 3 * 8 + 3);
  for (let epoch = 0; epoch < 3; epoch++) {
    for (const s of samples) {
      const f = forward(net, s.x);
      const g = backward(net, s.x, s.y, f);
      adamStep(net, state, g, 0.01);
    }
  }
  const finalLoss = loss(net);
  assert.ok(finalLoss < initial * 0.5, `loss ${initial} -> ${finalLoss}`);
});
