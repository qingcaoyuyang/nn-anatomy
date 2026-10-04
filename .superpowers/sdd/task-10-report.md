# Task 10 报告：神经网络舞台（核心可视化）

## 目标

实现课堂核心舞台：169 输入像素墙、16 隐藏、10 输出的三列网络图，静态权重连线 + 前向传导粒子动画 + 反向误差动画 + 权重热力图交互。

## 交付物

- `src/ui/stage.js`（约 230 行）：
  - 静态渲染：像素墙按激活强度着橙；每隐藏节点画 |w| top-12 连线（蓝正红负、粗细=|w|）；隐藏→输出每类 top-8
  - `playInference(x)`：JS 引擎 forward → 三阶段动画（输入→隐藏粒子 300ms、隐藏→输出粒子 300ms、输出概率从均匀 10% 收敛到 p）
  - `playTrainingStep(before, after)`：从输出反向发射红色误差粒子（误差=目标-预测，阈值 0.05 过滤）
  - `highlightNode(1, h)`：点击隐藏节点，像素墙区叠加 13×13 权重热力图（蓝正红负）
  - `prefers-reduced-motion` 下跳过动画瞬达结果
- `src/ui/particles.js`（约 90 行）：全局单一 rAF 循环驱动的沿边粒子系统，`fireEdge / wait / cancelAll`
- `src/main.js`：中央面板接入 stage 画布（900×560）；手写收笔自动播放推理；点击画布命中检测切热力图
- `src/ui/app.css`：`.stage-canvas` 自适应样式

## 设计决策

- 权重连线每节点只画 top-12/top-8：全量 2704 条连线在课堂投影上不可读，top-N 让"粗线=重要连接"直觉成立
- 动画粒子的大小/透明度编码信号强度（`value = 激活 × |w|`），强弱流量的视觉差异即"信号"概念
- 网络权重来自 seed=999 的教学引擎（与 Task 7 验收同一网络），未训练时输出接近均匀 10%，正好用于开场演示
- `outputBars` 收敛动画独立于粒子系统（需要逐帧重绘整图），结束后回落静态渲染

## 验证

- 模块导入冒烟：`stage: initStage`、`particles: cancelAll,fireEdge,wait` ✅
- Chrome headless DOM 渲染：`stage-canvas` 出现在文档 ✅；控制台无 JS 运行时错误 ✅
- `node --test`：11 passed / 0 failed（引擎层无回归）

## 遗留（计划内）

- `playTrainingStep` 已实现反向误差粒子，训练面板（Task 11）接入时与更新前后概率对比条联动
- 热力图点击交互需人工课堂验收（headless 无法模拟点击精度），留待 Task 12 端到端验收
