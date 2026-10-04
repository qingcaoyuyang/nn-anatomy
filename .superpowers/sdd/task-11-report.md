# Task 11 报告：训练控制面板 + 数据集管理

## 目标

打通 Rust command 层与前端：模型管理（新建/加载/重命名/删除）、训练控制（Adam/SGD 切换、对数 lr 滑条、单步/批量）、loss 曲线与准确率、数据集网格浏览与一键导入内置数据集。

## 交付物

### Rust 命令层（src-tauri/src/commands.rs，约 480 行）

- `AppState`：workspace + 当前模型 + Adam 状态 + epoch + 历史，thread_local 共享
- 模型：`model_create / model_load / model_rename / model_delete / model_list / model_import / model_export`
- 数据集：`dataset_add（base64 PNG）/ dataset_list（分页+缩略图）/ dataset_remove / dataset_stats / dataset_import_builtin（内置 1000/300 样本转 PNG 落盘）`
- 训练：`train_one_epoch / train_bulk / training_history / evaluate_test / training_reset`，每次训练后自动持久化到 `models/<name>.nnmodel.json`
- 自实现 base64 编解码（无额外依赖）

### 前端（src/ui/panel.js + dataset-viewer.js）

- 训练面板：模型下拉+新建、Adam/SGD 切换、lr 滑条（对数 0.0001-0.1）、epoch 加减（1-150）、单步训练/批量快进、loss 折线画布、训练/测试准确率芯片
- 数据集浏览器：train/test 切换、60 张分页网格（真实 PNG 缩略图 + 标签角标 + 删除）、一键导入按钮
- 保存到手写样本流程已预留（`dataset_add` command 就绪，画板"保存到训练集"按钮待 Task 12 接线）

## TDD 证据

1. base64 解码首轮发现 len=1/2 截断 bug（RED：roundtrip 断言失败）→ 修复为 2 字符=1 字节标准语义（GREEN）
2. PNG 往返测试：0.9→~230 灰度、0.5→~128 灰度，量化误差 < 1
3. 端到端：工作区导入 50/20 样本 → train_epoch → evaluate，loss 有限、acc ∈ [0,1]

## 测试结果

- cargo test --lib: **28 passed / 0 failed**（52s，含内置数据集验收测试）
- node --test: **11 passed / 0 failed**
- Chrome headless DOM：`ds-grid / model-select / loss-canvas / stage-canvas / train-controls` 全部渲染 ✅

## 已知限制（记录在案）

- main.rs 的 invoke_handler 尚未挂载新 commands（bin target 链接慢，集中到 Task 12 一次接线验证）
- 混淆矩阵 UI、画板保存按钮接线 → Task 12
- 模型切换后 JS 教学引擎的 net 需要同步（`model_export` 返回权重供 stage 重建）→ Task 12
