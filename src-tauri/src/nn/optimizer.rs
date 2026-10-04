use crate::nn::network::{Gradients, Network};

/// Adam moments over the flat parameter vector [w1, b1, w2, b2].
#[derive(Clone)]
pub struct AdamState {
    pub m: Vec<f64>,
    pub v: Vec<f64>,
    pub t: u64,
}

impl AdamState {
    pub fn new(param_count: usize) -> AdamState {
        AdamState { m: vec![0.0; param_count], v: vec![0.0; param_count], t: 0 }
    }
}

/// One Adam step. Constants are fixed on purpose so the JS teaching engine can
/// mirror every line and stay within 1e-9 of this implementation.
pub fn adam_step(net: &mut Network, state: &mut AdamState, grads: &Gradients, lr: f64) {
    const BETA1: f64 = 0.9;
    const BETA2: f64 = 0.999;
    const EPS: f64 = 1e-8;

    let flat_grads = flatten_grads(grads);
    let mut flat_params = flatten_params(net);
    assert_eq!(flat_params.len(), state.m.len(), "AdamState size must match parameter count");

    state.t += 1;
    let t = state.t as f64;
    for i in 0..flat_params.len() {
        let g = flat_grads[i];
        state.m[i] = BETA1 * state.m[i] + (1.0 - BETA1) * g;
        state.v[i] = BETA2 * state.v[i] + (1.0 - BETA2) * g * g;
        // Bias correction: m and v start at 0, so early estimates are scaled
        // down by (1 - beta^t); dividing by that recovers the true moments.
        let m_hat = state.m[i] / (1.0 - BETA1.powf(t));
        let v_hat = state.v[i] / (1.0 - BETA2.powf(t));
        flat_params[i] -= lr * m_hat / (v_hat.sqrt() + EPS);
    }
    unflatten_params(net, &flat_params);
}

pub fn sgd_step(net: &mut Network, grads: &Gradients, lr: f64) {
    let flat_grads = flatten_grads(grads);
    let mut flat_params = flatten_params(net);
    for i in 0..flat_params.len() {
        flat_params[i] -= lr * flat_grads[i];
    }
    unflatten_params(net, &flat_params);
}

/// Flat layout: [w1, b1, w2, b2] (row-major w1, then w2).
fn flatten_params(net: &Network) -> Vec<f64> {
    let mut out = Vec::with_capacity(
        net.w1.len() + net.b1.len() + net.w2.len() + net.b2.len(),
    );
    out.extend_from_slice(&net.w1);
    out.extend_from_slice(&net.b1);
    out.extend_from_slice(&net.w2);
    out.extend_from_slice(&net.b2);
    out
}

fn flatten_grads(grads: &Gradients) -> Vec<f64> {
    let mut out = Vec::with_capacity(
        grads.g_w1.len() + grads.g_b1.len() + grads.g_w2.len() + grads.g_b2.len(),
    );
    out.extend_from_slice(&grads.g_w1);
    out.extend_from_slice(&grads.g_b1);
    out.extend_from_slice(&grads.g_w2);
    out.extend_from_slice(&grads.g_b2);
    out
}

fn unflatten_params(net: &mut Network, flat: &[f64]) {
    let n1 = net.w1.len();
    let n2 = n1 + net.b1.len();
    let n3 = n2 + net.w2.len();
    let n4 = n3 + net.b2.len();
    net.w1.copy_from_slice(&flat[0..n1]);
    net.b1.copy_from_slice(&flat[n1..n2]);
    net.w2.copy_from_slice(&flat[n2..n3]);
    net.b2.copy_from_slice(&flat[n3..n4]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::network::{Arch, Gradients, Network};

    fn tiny() -> Network {
        Network::new(Arch { inputs: 3, hidden: 2, outputs: 2 }, 1)
    }

    fn param_count(net: &Network) -> usize {
        let Arch { inputs, hidden, outputs } = net.arch;
        hidden * inputs + hidden + outputs * hidden + outputs
    }

    fn grads_of(net: &Network, g1: f64, gb1: f64, g2: f64, gb2: f64) -> Gradients {
        let Arch { inputs, hidden, outputs } = net.arch;
        Gradients {
            g_w1: vec![g1; hidden * inputs],
            g_b1: vec![gb1; hidden],
            g_w2: vec![g2; outputs * hidden],
            g_b2: vec![gb2; outputs],
        }
    }

    #[test]
    fn adam_first_step_moves_each_param_by_about_lr() {
        let mut net = tiny();
        let before = net.clone();
        let grads = grads_of(&net, 0.5, -0.25, 1.0, -2.0);
        let mut state = AdamState::new(param_count(&net));
        let lr = 0.1;
        adam_step(&mut net, &mut state, &grads, lr);

        assert_eq!(state.t, 1);
        // After one step m_hat = g and v_hat = g^2 exactly, so the normalized
        // update is g / (|g| + eps) ~ sign(g): |delta| ~ lr, sign opposite to g.
        let eps_scale = 1e-6;
        for (i, (&g, &b)) in grads.g_w1.iter().zip(before.w1.iter()).enumerate() {
            let delta = net.w1[i] - b;
            assert!((delta.abs() - lr).abs() < eps_scale, "w1[{}]: |{}| != lr", i, delta);
            assert!((delta + lr * g.signum()).abs() < eps_scale);
        }
        for i in 0..before.b1.len() {
            let delta = net.b1[i] - before.b1[i];
            assert!((delta.abs() - lr).abs() < eps_scale);
        }
        for i in 0..before.w2.len() {
            let delta = net.w2[i] - before.w2[i];
            assert!((delta.abs() - lr).abs() < eps_scale);
        }
        for i in 0..before.b2.len() {
            let delta = net.b2[i] - before.b2[i];
            assert!((delta.abs() - lr).abs() < eps_scale);
        }
        // Moments consumed: m1 = (1-beta1)*g, v1 = (1-beta2)*g^2.
        assert!((state.m[0] - 0.1 * 0.5).abs() < 1e-15);
        assert!((state.v[0] - 0.001 * 0.25).abs() < 1e-15);
        assert_eq!(state.m.len(), param_count(&net));
        assert_eq!(state.v.len(), param_count(&net));
    }

    #[test]
    fn sgd_step_is_exact_negative_lr_times_grad() {
        let mut net = tiny();
        let before = net.clone();
        let grads = grads_of(&net, 0.25, -0.5, 1.5, -2.0);
        let lr = 0.3;
        sgd_step(&mut net, &grads, lr);
        for i in 0..before.w1.len() {
            assert_eq!(net.w1[i], before.w1[i] - lr * grads.g_w1[i]);
        }
        for i in 0..before.b1.len() {
            assert_eq!(net.b1[i], before.b1[i] - lr * grads.g_b1[i]);
        }
        for i in 0..before.w2.len() {
            assert_eq!(net.w2[i], before.w2[i] - lr * grads.g_w2[i]);
        }
        for i in 0..before.b2.len() {
            assert_eq!(net.b2[i], before.b2[i] - lr * grads.g_b2[i]);
        }
    }

    #[test]
    fn adam_training_reduces_loss_over_100_steps() {
        let mut net = Network::new(Arch { inputs: 4, hidden: 8, outputs: 3 }, 9);
        let x = vec![0.2, -0.4, 0.6, 0.1];
        let y = 1usize;
        let loss = |net: &Network| -net.forward(&x).p[y].ln();
        let initial = loss(&net);
        let mut state = AdamState::new(param_count(&net));
        for _ in 0..100 {
            let f = net.forward(&x);
            let g = net.backward(&x, y, &f);
            adam_step(&mut net, &mut state, &g, 0.05);
        }
        let final_loss = loss(&net);
        assert!(
            final_loss < 0.5 * initial,
            "loss did not converge: {} -> {}",
            initial,
            final_loss
        );
    }
}
