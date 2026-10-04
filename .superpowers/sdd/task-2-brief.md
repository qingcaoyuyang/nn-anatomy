### Task 2: Rust 网络核心（前向/反向/初始化）

**Files:**
- Create: src-tauri/src/nn/mod.rs
- Create: src-tauri/src/nn/network.rs
- Test: src-tauri/src/nn/network.rs 内联 #[cfg(test)]

**Interfaces:**
- Produces: `Network { arch: Arch, w1: Vec<f64>, b1: Vec<f64>, w2: Vec<f64>, b2: Vec<f64> }`
- Produces: `Network::new(arch: Arch, seed: u64) -> Network`（He 初始化）
- Produces: `Network::forward(&self, x: &[f64]) -> ForwardOut { z1, a1, z2, p }`
- Produces: `Network::backward(&self, x: &[f64], y: usize, f: &ForwardOut) -> Gradients`

- [ ] **Step 1: 写失败测试（含数值微分校验）**

```rust
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
        // 校验 W1[3*169+10]、W2[4]、B1[5]
        let cases = [
            (0, 3 * 169 + 10, g.g_w1[3 * 169 + 10]),
            (1, 4, g.g_w2[4]),
            (2, 5, g.g_b1[5]),
        ];
        for (which, idx, ana) in cases {
            let mut up = net.clone(); let mut dn = net.clone();
            match which { 0 => { up.w1[idx] += eps; dn.w1[idx] -= eps; }
                          1 => { up.w2[idx] += eps; dn.w2[idx] -= eps; }
                          _ => { up.b1[idx] += eps; dn.b1[idx] -= eps; } }
            let lp = up.forward(&x).p[y].ln();
            let ld = dn.forward(&x).p[y].ln();
            let num = (lp - ld) / (2.0 * eps);
            assert!((num - ana).abs() < 1e-6, "梯度不匹配: {} vs {}", num, ana);
        }
    }
}
```

- [ ] **Step 2: cargo test 确认失败**

Run: cd src-tauri && cargo test

- [ ] **Step 3: 实现 network.rs**

```rust
use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct Arch { pub inputs: usize, pub hidden: usize, pub outputs: usize }

#[derive(Clone, Serialize, Deserialize)]
pub struct Network { pub arch: Arch, pub w1: Vec<f64>, pub b1: Vec<f64>, pub w2: Vec<f64>, pub b2: Vec<f64> }

#[derive(Clone)]
pub struct ForwardOut { pub z1: Vec<f64>, pub a1: Vec<f64>, pub z2: Vec<f64>, pub p: Vec<f64> }

#[derive(Clone)]
pub struct Gradients { pub g_w1: Vec<f64>, pub g_b1: Vec<f64>, pub g_w2: Vec<f64>, pub g_b2: Vec<f64> }
```
（实现 new/forward/backward：LeakyReLU + Softmax + 交叉熵，权重 He 初始化，RNG 用手写 mulberry32）

- [ ] **Step 4: cargo test 确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: rust network core"**

