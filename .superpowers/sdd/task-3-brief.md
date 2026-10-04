### Task 3: Rust 优化器（Adam + SGD）

**Files:**
- Create: src-tauri/src/nn/optimizer.rs
- Test: 内联 #[cfg(test)]

**Interfaces:**
- Consumes: Network/Gradients from Task 2
- Produces: `AdamState { m: Vec<f64>, v: Vec<f64>, t: u64 }`
- Produces: `AdamState::new(param_count: usize)`
- Produces: `adam_step(net: &mut Network, state: &mut AdamState, grads: &Gradients, lr: f64)`
- Produces: `sgd_step(net: &mut Network, grads: &Gradients, lr: f64)`

- [ ] **Step 1: 失败测试**：Adam 第一步 |Δw| ≈ lr（偏差校正后）；SGD Δw = lr*grad；100 步 loss 下降。
- [ ] **Step 2: cargo test 确认失败**
- [ ] **Step 3: 实现**：m/v 一阶二阶矩、t 步数、偏差校正；参数扁平化布局 w1+b1+w2+b2 拼接。
- [ ] **Step 4: cargo test 确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: rust optimizers"**

