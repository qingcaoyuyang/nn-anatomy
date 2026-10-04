### Task 4: 预处理（Rust + JS 双实现）

**Files:**
- Create: src-tauri/src/nn/preprocess.rs
- Create: src/nn/preprocess.js
- Test: 双方各写测试（cargo test / node --test）

**Interfaces:**
- Produces (Rust): `center_of_mass(img: &[u8], w: usize, h: usize) -> (f64, f64)`
- Produces (Rust): `preprocess(img: &[u8], w: usize, h: usize) -> Vec<f64>`（居中+降采样 13×13）
- Produces (JS): centerOfMass(img, w, h)、resampleTo13(img, w, h)、preprocess(img, w, h)

- [ ] **Step 1: 失败测试**（两语言同断言）：全零图像重心=画布中心；单亮点重心=亮点处；居中后降采样重心≈(6,6)；输出 0~1。
- [ ] **Step 2: 运行确认失败**
- [ ] **Step 3: 实现双份**（算法逐行对应：双线性采样 + 2×2 块平均）
- [ ] **Step 4: 运行确认通过**
- [ ] **Step 5: 提交 git commit -m "feat: dual-language preprocessing"**

