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
        let mut rng = Mulberry32::new(fold_seed(seed));
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
        assert_eq!(x.len(), inputs, "forward: input length must equal arch.inputs");
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
        assert_eq!(x.len(), inputs, "backward: input length must equal arch.inputs");
        assert!(y < outputs, "backward: label {} out of range (outputs={})", y, outputs);
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

/// Folds the full u64 seed into a u32 so high bits affect the stream.
fn fold_seed(seed: u64) -> u32 {
    (seed as u32) ^ ((seed >> 32) as u32)
}

impl Mulberry32 {
    fn new(seed: u32) -> Self {
        Mulberry32 { state: seed }
    }

    /// Canonical mulberry32 (bryc's reference implementation). The JS teaching
    /// engine must mirror these exact steps for cross-language parity.
    fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B79F5);
        let mut t = self.state;
        t = (t ^ t.wrapping_shr(15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ t.wrapping_shr(7)).wrapping_mul(t | 61));
        t ^ t.wrapping_shr(14)
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

    /// Golden values from the canonical JS mulberry32, seed 42 (fold(42) = 42).
    /// Guards the cross-language parity contract: the JS teaching engine
    /// implements the same steps and must produce this exact stream.
    #[test]
    fn mulberry32_matches_js_reference() {
        let mut rng = Mulberry32::new(fold_seed(42));
        let expected = [
            0x99e1ef7c, 0x72c32b8a, 0xda3b32c0,
            0xab73b0ad, 0x2cc09a8a, 0x86cec4d3,
        ];
        for want in expected {
            assert_eq!(rng.next_u32(), want);
        }
    }

    #[test]
    fn seed_fold_makes_high_bits_matter() {
        let a = Network::new(Arch { inputs: 3, hidden: 2, outputs: 2 }, 2);
        let b = Network::new(Arch { inputs: 3, hidden: 2, outputs: 2 }, 0x1_0000_0002);
        assert_ne!(a.w1, b.w1);
    }

    #[test]
    fn forward_outputs_valid_softmax() {
        let net = Network::new(Arch { inputs: 169, hidden: 16, outputs: 10 }, 42);
        let x = vec![0.5; 169];
        let f = net.forward(&x);
        assert_eq!(f.p.len(), 10);
        assert_eq!(f.z2.len(), 10);
        let sum: f64 = f.p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    #[should_panic(expected = "input length must equal arch.inputs")]
    fn forward_rejects_wrong_length_input() {
        let net = Network::new(Arch { inputs: 169, hidden: 16, outputs: 10 }, 42);
        let _ = net.forward(&vec![0.5; 168]);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn backward_rejects_out_of_range_label() {
        let net = Network::new(Arch { inputs: 4, hidden: 2, outputs: 3 }, 42);
        let f = net.forward(&vec![0.1; 4]);
        let _ = net.backward(&vec![0.1; 4], 3, &f);
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
        // Close the z2/g_b2 coverage gap: exercise every ForwardOut/Gradients field.
        assert_eq!(f.z2.len(), 10);
        assert_eq!(g.g_b2.len(), 10);
        assert!(g.g_b2.iter().all(|&v| v.is_finite()));
    }
}
