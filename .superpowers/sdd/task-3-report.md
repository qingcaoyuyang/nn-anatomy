### Task 3 Report: Rust 优化器（Adam + SGD）

## 交付内容

- 新建 `src-tauri/src/nn/optimizer.rs`：Adam（一阶矩 m / 二阶矩 v / 步数 t / 偏差校正）与 SGD 两个优化器，均以 `Network` 的 `Gradients` 为输入就地更新参数。
- `src-tauri/src/nn/mod.rs` 挂载 `pub mod optimizer`。
- 参数扁平化布局固定为 `w1 ++ b1 ++ w2 ++ b2`（w1/w2 均为行主序），与 JS 教学引擎镜像实现保持同一布局。
- Adam 常量写死在函数内：`BETA1 = 0.9`、`BETA2 = 0.999`、`EPS = 1e-8`；运算为逐参数单循环顺序执行，无并行/随机性，保证跨语言奇偶（误差 < 1e-9）可逐行翻译。

## 接口签名

```rust
pub struct AdamState { pub m: Vec<f64>, pub v: Vec<f64>, pub t: u64 }
impl AdamState { pub fn new(param_count: usize) -> AdamState }
pub fn adam_step(net: &mut Network, state: &mut AdamState, grads: &Gradients, lr: f64)
pub fn sgd_step(net: &mut Network, grads: &Gradients, lr: f64)
```

`param_count = hidden*inputs + hidden + outputs*hidden + outputs`，`AdamState::new` 传入该值即可；`adam_step` 内部会断言状态长度与参数数一致。

## TDD 证据

**RED**（先写三个测试，实现尚未存在）：`cargo test --lib` 编译失败，报 `AdamState`/`adam_step`/`sgd_step` 未定义：

```
error[E0433]: cannot find type `AdamState` in this scope
error[E0425]: cannot find function `adam_step` in this scope
error[E0425]: cannot find function `sgd_step` in this scope
error: could not compile `nn-anatomy` (lib test) due to 5 previous errors; 1 warning emitted
```

**GREEN**（实现后）：

```
running 9 tests
test nn::optimizer::tests::adam_first_step_moves_each_param_by_about_lr ... ok
test nn::optimizer::tests::sgd_step_is_exact_negative_lr_times_grad ... ok
test nn::optimizer::tests::adam_training_reduces_loss_over_100_steps ... ok
（另有 6 个 Task 2 既有测试，全部 ok）
test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

三个测试场景对应 brief Step 1：

- Adam 第一步：`|Δw| ≈ lr`（第 1 步时 `m_hat = g`、`v_hat = g²`，偏差校正后归一化更新 ≈ sign(g)），并同时验证 m/v/t 与扁平长度。
- SGD：`Δw = -lr * grad` 精确断言（`assert_eq!`）。
- 收敛：固定样本 100 步 Adam 训练，最终 loss < 初始 loss 的 50%。

验证命令：`cd src-tauri && source ~/.cargo/env && cargo test --lib`（全绿，无编译警告，全程 < 1 秒）。

## Commit

提交信息：`feat: rust optimizers`，哈希：`694de0f`（包含本报告、Task 3 brief、optimizer.rs 及 Task 2 遗留评审 diff）。
