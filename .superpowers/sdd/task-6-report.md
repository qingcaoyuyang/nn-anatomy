### Task 6 Report: 工作区文件系统

## 交付内容

- `src-tauri/src/workspace.rs`：
  - `Workspace::open(path)`（校验 workspace.json 存在）/ `Workspace::create(path)`（幂等创建 datasets/train|test/0-9、models、exports 目录树）
  - `add_sample`：PNG 写入 datasets/&lt;split&gt;/&lt;label&gt;/&lt;id&gt;.png + manifest 记录（id、label、file、source、createdAt、width/height）
  - `remove_sample`：删除 PNG + manifest 条目
  - `list_samples`：按 createdAt 排序返回
  - `save_model` / `load_model` / `list_models` / `delete_model`：models/&lt;name&gt;.nnmodel.json 文本格式（serde_json pretty）
  - `read_sample_png`：训练/推理时读原始 PNG
  - id 格式：s&lt;毫秒时间戳&gt;-&lt;进程内计数器&gt;
- `src-tauri/src/lib.rs` 挂载 workspace 模块
- `nn/network.rs` 补充 Arch/Network 的 Debug derive（测试断言需要）

## 测试（tempdir 隔离）

1. `workspace_roundtrip_add_list_remove`：手写 260×260 入训练集、MNIST 13×13 入测试集 → 文件落盘路径正确 → 重新 open 后 list 读回字段全对 → remove 后 PNG 与条目都消失
2. `model_save_load_delete_roundtrip`：中文文件名模型保存/加载/列举/删除
3. `create_is_idempotent_and_open_fails_on_missing`：重复 create 不报错；open 非工作区目录报错

**RED**: 17 个编译错误（Workspace/Split/Source/ModelFile 未定义）
**GREEN**: `cargo test --lib` 18 passed / 0 failed（含此前全部任务）

## Commit

`feat: workspace filesystem`
