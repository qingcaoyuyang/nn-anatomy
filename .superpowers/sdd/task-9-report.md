# Task 9 报告：UI 骨架 + 手写画板

## 目标

搭建教学界面骨架：三栏布局（采集 / 网络舞台 / 训练面板）、页头阶段标签、手写画板与 13×13 实时预览，为 Task 10/11 的舞台动效与训练面板就位。

## 交付物

- `src/main.js`（130 行）：无框架原生 ESM 渲染三栏骨架；数字标注按钮 0~9；保存到训练集/测试集按钮（占位）；13×13 预览渲染
- `src/ui/drawing.js`（85 行）：`initDrawing(canvas, {lineWidth, onStroke})`，pointer events + setPointerCapture（触屏兼容），每次落笔/收笔后 `getImageData → 灰度 → preprocess()` 回调，暴露 `clear() / getPixels13() / isBlank()`
- `src/ui/app.css`（189 行）：设计令牌（暖橙 #F97316 单强调色、#FFF7ED 背景、8px 圆角、仅浅色）、300px/1fr/320px 三栏 grid、平板 1024px 双栏与 768px 单列响应式、页面载入 rise 动画

## 设计决策

- 画板内部 260×260（1 格 = 20px 的 13 倍数），保证与 MNIST 13×13 样本同一预处理管线（居中 + 双线性重采样）
- 预览每格用橙色 alpha 表达激活强度，让学员直观看到"网络看到的不是笔画而是像素强度"
- `touch-action: none` 仅作用于画板，平板书写不触发页面滚动
- 字体按 Task 1 评审遗留留待打包（当前 fallback 到系统 PingFang SC）

## 验证

- `node --test`：11 passed / 0 failed（JS 引擎层无回归）
- `cargo check --lib`：通过，仅既有 dead_code 警告（workspace.rs from_str）
- bin target 全量链接在 16GB 机器上慢为已知问题（Task 2 已记录），留待 final review 用 `CARGO_INCREMENTAL=0` 复验

## 遗留（计划内）

- 保存按钮的 Tauri command 接线、阶段标签切换逻辑 → Task 10/11 随舞台与训练面板实现
- 网络舞台与训练面板当前为斜纹占位区，视觉基调已就位
