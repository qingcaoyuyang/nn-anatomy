### Task 5: JS 教学引擎（前向/反向/Adam）与双引擎一致性

**Files:**
- Create: src/nn/network.js
- Create: src/nn/optimizer.js
- Create: tests/fixtures/（交叉验证权重 JSON）
- Test: tests/network.test.js、tests/optimizer.test.js

**Interfaces:**
- Produces: createNetwork(arch, seed) → {arch, W1, b1, W2, b2}
- Produces: forward(net, x) → {z1, a1, z2, p}
- Produces: backward(net, x, y, f) → {gW1, gb1, gW2, gb2}
- Produces: createAdam() / adamStep(net, state, grads, lr) / sgdStep(net, grads, lr)

- [ ] **Step 1: 写失败测试**（含 100 样本 3 epoch loss 下降测试）
- [ ] **Step 2: node --test 确认失败**
- [ ] **Step 3: 实现**（与 Rust 逻辑严格对应：同 seed 生成同权重，mulberry32+Box-Muller）
- [ ] **Step 4: node --test 确认通过**
- [ ] **Step 4b: 双引擎一致性测试**：固定 seed=42、固定 20 样本，JS 训练 3 epoch 导出权重至 tests/fixtures/js-weights.json；Rust 同 seed 同样本训练 3 epoch，逐元素比较 |Δw| < 1e-9（cargo test 读取 fixture 文件）
- [ ] **Step 5: 提交 git commit -m "feat: js teaching engine with cross-validation"**

