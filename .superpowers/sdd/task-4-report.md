### Task 4 Report: 预处理（Rust + JS 双实现）

## 交付内容

- `src-tauri/src/nn/preprocess.rs`：`center_of_mass`、`center_of_mass_f64`、`preprocess`（居中 + 双线性重采样 13×13 + 归一化）
- `src/nn/preprocess.js`：`centerOfMass`、`resampleTo13(img, w, h, dx, dy)`、`preprocess`（逐行镜像 Rust 实现）
- `src/nn/preprocess.test.js`：5 个测试（4 个行为测试 + 1 个跨语言奇偶黄金值测试）
- `src-tauri/src/nn/mod.rs` 挂载 `pub mod preprocess`

## 算法

1. 重心：强度加权质心，空图返回画布中心（保证居中是 no-op）
2. 平移：dx = (w-1)/2 - cx，dy 同理，把重心移到画布中心
3. 重采样：13×13 每个输出格中心映射回源坐标（减去平移量），双线性采样，边缘外读零
4. 归一化：除以 255，输出 [0,1]

## TDD 证据

**RED**（双语言）：

- Rust: `cargo test --lib` 5 个编译错误（`center_of_mass`/`preprocess`/`center_of_mass_f64` 未定义）
- JS: `node --test` 失败（`./preprocess.js` 模块不存在）

**GREEN**（双语言）：

- Rust: `cargo test --lib` → 14 passed / 0 failed（Task 2/3 的 10 个 + 本任务 4 个）
- JS: `node --test src/nn/preprocess.test.js` → 5 passed / 0 failed

## 跨语言奇偶

黄金值由 Rust 实现生成（`{:e}` 全精度打印，169 值 × 2 个用例：5×5 对角线笔画、26×26 径向渐变圆），JS 测试断言逐值误差 < 1e-9。Rust 侧另有 `golden_values_parity_with_js` 测试锁定关键值，双向防漂移。5×5 对角图形具有 180° 旋转对称性，末格值镜像首格（out[168] ≈ out[0]），测试已覆盖。

## Commit

`feat: dual-language preprocessing`
