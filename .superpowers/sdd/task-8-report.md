# Task 8 报告：Rust 训练引擎

## 目标

把 Network/Adam/Gradients 组装成可复用的训练循环与评估器，为 Tauri command 层和前端训练面板提供核心 API。

## 交付物（src-tauri/src/trainer.rs）

- `LabeledSample {x, y}`：预处理后的训练样本
- `train_epoch(net, adam, samples, lr, epoch, use_sgd) -> EpochReport{loss, acc}`：LCG 确定性 shuffle，逐样本前向→反向→Adam/SGD 步进
- `evaluate(net, samples) -> EvalReport{acc, per_class, confusion}`：10×10 混淆矩阵 + 每类正确率
- `cross_entropy_loss` / `forward_full`：供推理展示返回全部中间值

## TDD 证据

1. RED：先写 4 个测试（loss 下降、confusion 对角线、Adam/SGD 双路径、内置数据集 90%）
2. 首轮编译失败（类型错误）修复后 3 个单元测试通过
3. GREEN：全部 25 个 Rust 测试通过（含 90% 验收 22s）

## 测试结果

- cargo test --lib: 25 passed / 0 failed
- node --test: 11 passed / 0 failed

## 决策

- shuffle 逻辑与 sample_data.rs 完全一致（epoch*黄金比+42 种子），保证 Task 7 验收结果可复现
- train_bulk 命令层（异步/取消/状态机）留给 Task 9 UI 集成时一并实现，避免无 UI 的死代码
