use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Arch {
    pub inputs: usize,
    pub hidden: usize,
    pub outputs: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub arch: Arch,
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

#[derive(Clone)]
pub struct ForwardOut {
    pub z1: Vec<f64>,
    pub a1: Vec<f64>,
    pub z2: Vec<f64>,
    pub p: Vec<f64>,
}

#[derive(Clone)]
pub struct Gradients {
    pub g_w1: Vec<f64>,
    pub g_b1: Vec<f64>,
    pub g_w2: Vec<f64>,
    pub g_b2: Vec<f64>,
}

impl Network {
    /// He-initialized MLP: W ~ N(0, sqrt(2/fan_in)) via mulberry32 + Box-Muller.
    pub fn new(arch: Arch, seed: u64) -> Network {
        let mut rng = Mulberry32::new(seed);
        let std1 = (2.0 / arch.inputs as f64).sqrt();
        let std2 = (2.0 / arch.hidden as f64).sqrt();
        let w1: Vec<f64> = (0..arch.hidden * arch.inputs)
            .map(|_| rng.gaussian() * std1)
            .collect();
        let b1 = vec![0.0; arch.hidden];
        let w2: Vec<f64> = (0..arch.outputs * arch.hidden)
            .map(|_| rng.gaussian() * std2)
            .collect();
        let b2 = vec![0.0; arch.outputs];
        Network { arch, w1, b1, w2, b2 }
    }

    pub fn forward(&self, x: &[f64]) -> ForwardOut {
        let Arch { inputs, hidden, outputs } = self.arch;
        // z1 = W1 x + b1 (row-major [hidden x inputs]), a1 = LeakyReLU(0.01)
        let mut z1 = self.b1.clone();
        for h in 0..hidden {
            let row = &self.w1[h * inputs..(h + 1) * inputs];
            let mut acc = 0.0;
            for (j, &xv) in x.iter().enumerate() {
                acc += row[j] * xv;
            }
            z1[h] += acc;
        }
        let a1: Vec<f64> = z1.iter().map(|&z| leaky_relu(z)).collect();
        // z2 = W2 a1 + b2 (row-major [outputs x hidden]), p = softmax(z2)
        let mut z2 = self.b2.clone();
        for o in 0..outputs {
            let row = &self.w2[o * hidden..(o + 1) * hidden];
            let mut acc = 0.0;
            for (h, &av) in a1.iter().enumerate() {
                acc += row[h] * av;
            }
            z2[o] += acc;
        }
        let p = softmax(&z2);
        ForwardOut { z1, a1, z2, p }
    }

    /// Gradients of L = -ln p[y] w.r.t. all parameters.
    pub fn backward(&self, x: &[f64], y: usize, f: &ForwardOut) -> Gradients {
        let Arch { inputs, hidden, outputs } = self.arch;
        // dL/dz2 = p - onehot(y)
        let dz2: Vec<f64> = (0..outputs)
            .map(|o| f.p[o] - if o == y { 1.0 } else { 0.0 })
            .collect();
        let g_w2: Vec<f64> = (0..outputs * hidden)
            .map(|i| dz2[i / hidden] * f.a1[i % hidden])
            .collect();
        let g_b2 = dz2.clone();
        // backprop through LeakyReLU into hidden layer
        let da1: Vec<f64> = (0..hidden)
            .map(|h| {
                (0..outputs).map(|o| self.w2[o * hidden + h] * dz2[o]).sum::<f64>()
            })
            .collect();
        let dz1: Vec<f64> = (0..hidden)
            .map(|h| da1[h] * if f.z1[h] > 0.0 { 1.0 } else { 0.01 })
            .collect();
        let g_w1: Vec<f64> = (0..hidden * inputs)
            .map(|i| dz1[i / inputs] * x[i % inputs])
            .collect();
        let g_b1 = dz1;
        Gradients { g_w1, g_b1, g_w2, g_b2 }
    }
}

fn leaky_relu(z: f64) -> f64 {
    if z > 0.0 { z } else { 0.01 * z }
}

/// Softmax with max-shift for numerical stability.
fn softmax(z: &[f64]) -> Vec<f64> {
    let max = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = z.iter().map(|&v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|&e| e / sum).collect()
}

struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    fn new(seed: u64) -> Self {
        Mulberry32 { state: seed as u32 }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B79F5);
        let mut t = self.state;
        t = t.wrapping_mul(t.wrapping_add(0x1D872B41) ^ t.wrapping_shr(15));
        t ^= t.wrapping_shl(7).wrapping_mul(0x2B7E1516);
        t = t.wrapping_shr(15) ^ t;
        t
    }

    /// Uniform in (0, 1), avoiding exact 0/2^32 endpoints for Box-Muller.
    fn uniform(&mut self) -> f64 {
        (self.next_u32() >> 8) as f64 / (1u64 << 24) as f64
    }

    fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_outputs_valid_softmax() {
        let net = Network::new(Arch { inputs: 169, hidden: 16, outputs: 10 }, 42);
        let x = vec![0.5; 169];
        let f = net.forward(&x);
        assert_eq!(f.p.len(), 10);
        let sum: f64 = f.p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn backward_matches_numerical_gradient() {
        let net = Network::new(Arch { inputs: 169, hidden: 16, outputs: 10 }, 7);
        let x: Vec<f64> = (0..169).map(|i| ((i * 37) % 13) as f64 / 13.0).collect();
        let y = 3;
        let f = net.forward(&x);
        let g = net.backward(&x, y, &f);
        let eps = 1e-5;
        // Numerical gradient check via central differences on L = -ln p[y].
        let cases = [
            (0, 3 * 169 + 10, g.g_w1[3 * 169 + 10]),
            (1, 4, g.g_w2[4]),
            (2, 5, g.g_b1[5]),
        ];
        for (which, idx, ana) in cases {
            let mut up = net.clone();
            let mut dn = net.clone();
            match which {
                0 => { up.w1[idx] += eps; dn.w1[idx] -= eps; }
                1 => { up.w2[idx] += eps; dn.w2[idx] -= eps; }
                _ => { up.b1[idx] += eps; dn.b1[idx] -= eps; }
            }
            let lp = -up.forward(&x).p[y].ln();
            let ld = -dn.forward(&x).p[y].ln();
            let num = (lp - ld) / (2.0 * eps);
            assert!((num - ana).abs() < 1e-6, "gradient mismatch: {} vs {}", num, ana);
        }
    }
}
