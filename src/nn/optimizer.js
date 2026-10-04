/**
 * Adam and SGD optimizers mirroring src-tauri/src/nn/optimizer.rs line for
 * line. Flat parameter layout: W1 ++ b1 ++ W2 ++ b2. Parity error < 1e-9.
 */

export function createAdam(paramCount) {
  return { m: new Array(paramCount).fill(0), v: new Array(paramCount).fill(0), t: 0 };
}

function flattenParams(net) {
  return [...net.W1, ...net.b1, ...net.W2, ...net.b2];
}

function flattenGrads(grads) {
  return [...grads.gW1, ...grads.gb1, ...grads.gW2, ...grads.gb2];
}

function unflattenParams(net, flat) {
  let n = 0;
  for (let i = 0; i < net.W1.length; i++) net.W1[i] = flat[n++];
  for (let i = 0; i < net.b1.length; i++) net.b1[i] = flat[n++];
  for (let i = 0; i < net.W2.length; i++) net.W2[i] = flat[n++];
  for (let i = 0; i < net.b2.length; i++) net.b2[i] = flat[n++];
}

export function adamStep(net, state, grads, lr) {
  const BETA1 = 0.9;
  const BETA2 = 0.999;
  const EPS = 1e-8;
  const flatGrads = flattenGrads(grads);
  const flatParams = flattenParams(net);
  if (flatParams.length !== state.m.length) {
    throw new Error("AdamState size must match parameter count");
  }
  state.t += 1;
  const t = state.t;
  for (let i = 0; i < flatParams.length; i++) {
    const g = flatGrads[i];
    state.m[i] = BETA1 * state.m[i] + (1 - BETA1) * g;
    state.v[i] = BETA2 * state.v[i] + (1 - BETA2) * g * g;
    const mHat = state.m[i] / (1 - Math.pow(BETA1, t));
    const vHat = state.v[i] / (1 - Math.pow(BETA2, t));
    flatParams[i] -= (lr * mHat) / (Math.sqrt(vHat) + EPS);
  }
  unflattenParams(net, flatParams);
}

export function sgdStep(net, grads, lr) {
  const flatGrads = flattenGrads(grads);
  const flatParams = flattenParams(net);
  for (let i = 0; i < flatParams.length; i++) {
    flatParams[i] -= lr * flatGrads[i];
  }
  unflattenParams(net, flatParams);
}
