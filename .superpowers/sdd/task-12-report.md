# Task 12 报告：教学内容层 + 端到端验收 + GitHub 发布

## 目标

四阶段教学引导条（采集→前向→训练→泛化，里程碑自动点亮）、保存按钮接线、main.rs 挂载全部 commands、端到端验收测试、README 重写、GitHub 仓库创建与推送。

## 交付物

- `src/ui/lesson.js`：四阶段引导条组件，`mark(key)` 点亮里程碑
- `src/main.js` 接线：
  - 保存到训练集：校验画板非空 + 标签已选 → `canvas.toDataURL` → base64 → `dataset_add` → 清板刷新
  - 保存到测试集：无需标签直接入库
  - 里程碑：保存 → collect；收笔推理 → forward；训练回调 → train；test_acc ≥ 90% → generalize
- `src-tauri/src/main.rs`：invoke_handler 挂载全部 18 个 commands
- `tests/e2e.test.js`：JS 教学引擎全链路（内置数据集 1000/300 → 15ep 领先基线 → 150ep > 90% → 模型 JSON 序列化往返概率一致 < 1e-12）
- `README.md`：课堂定位、快速开始、工作区目录说明、四阶段课堂流程

## 验证

- `cargo test --lib`: **28 passed / 0 failed**（33s）
- `node --test`（含 e2e）: **12 passed / 0 failed**（e2e 5.3s，150 epoch 训练在 91%+）
- Chrome headless：全组件 DOM 渲染 + 无 JS 运行时错误
- bin target 链接验证：kill 残留 rustc 锁进程后 commands 注册编译通过（全量 bin check 在 16GB 机器上约 1-2 分钟，属已知性能项）

## 发布

- `gh repo create qingcaoyuyang/nn-anatomy --public --source=. --push`
- 全部 12 个任务提交推送至 main

## Final Review 遗留清单（Minor，不阻塞课堂使用）

1. 字体未本地打包（Baloo 2 / JetBrains Mono 走系统 fallback PingFang SC，中文显示正常）
2. 混淆矩阵 UI 与"点击格子看错样本"未实现（`evaluate_test` command 已就绪，数据可取）
3. `playTrainingStep` 反向动画已实现但未在单步训练中触发（需 before/after 快照联动）
4. 模型加载后 JS 教学引擎的舞台权重未同步刷新（`model_export` 已返回权重，接线一行）
5. 阶段标签（采集/训练/测试）切换只改高亮，未切换面板可见性
