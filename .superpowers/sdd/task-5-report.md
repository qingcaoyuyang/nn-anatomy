### Task 5 Report: JS 教学引擎与双引擎一致性

## 交付内容

- `src/nn/network.js`：`createNetwork(arch, seed)`、`forward(net, x)`、`backward(net, x, y, f)`——镜像 Rust network.rs 逐行（mulberry32 + foldSeed + Box-Muller 同运算顺序、LeakyReLU、max-shift softmax、行主序矩阵乘、梯度链式法则）
- `src/nn/optimizer.js`：`createAdam(paramCount)`、`adamStep(net, state, grads, lr)`、`sgdStep(net, grads, lr)`——镜像 Rust optimizer.rs（扁平布局 W1++b1++W2++b2、beta1=0.9/beta2=0.999/eps=1e-8、偏差校正）
- `tests/network.test.js`（3 测试）、`tests/optimizer.test.js`（3 测试）
- `tests/fixtures/js-weights.json`：JS 训练产物（seed 42、20 样本、3 epoch、lr 0.01）
- Rust 侧新增 `rust_weights_match_js_fixture` 一致性测试

## TDD 证据

**RED**: node --test 两文件失败（`../src/nn/network.js`/`optimizer.js` 模块不存在）

**GREEN**: JS 6 passed；随后 Step 4b 一致性：Rust 读取 fixture 重建同训练循环，`rust_weights_match_js_fixture` ok——**逐元素误差 < 1e-9**。

全量：Rust `cargo test --lib` 15 passed / 0 failed；JS `node --test` 11 passed / 0 failed。

## 一致性方法

样本规则固定（20 个 4 维向量、标签 i%3、特征按类推进）、seed=42、Adam lr=0.01、3 epoch、样本顺序一致。JS 导出全精度 JSON，Rust 用 serde_json 读回逐元素 assert。fixture 进入版本库，未来任何一侧改动破坏奇偶都会被 CI 捕获。

## Commit

`feat: js teaching engine with cross-validation`
