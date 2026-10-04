### Task 8: 训练引擎（Rust command 层）

**Files:**
- Create: src-tauri/src/trainer.rs
- Test: 内联 #[cfg(test)]

**Interfaces:**
- Consumes: Network/Adam/Gradients/preprocess/sample-dataset
- Produces: `train_epochs(net, adam, samples, epochs, lr, on_epoch)`
- Produces: `evaluate(net, samples) -> { acc, per_class, confusion }`
- Produces commands: `train_bulk(model_name, epochs, lr, optimizer)`（异步、可取消）、`infer_13(x13: Vec<f64>)`（返回全部中间值）

- [ ] **Step 1: 失败测试**：100 样本 3 epoch loss 下降；confusion 10×10 对角线正确；train_bulk 取消后保留已完成 epoch。
- [ ] **Step 2: cargo test 确认失败**
- [ ] **Step 3: 实现**：每轮 shuffle（seed 固定）、每 epoch 回调；train_bulk 状态机 Mutex<TrainingState>。
- [ ] **Step 4: cargo test 确认通过（含 1000 样本 15 epoch acc>0.9）**
- [ ] **Step 5: 提交 git commit -m "feat: rust training engine"**

