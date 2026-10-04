# nn-anatomy · 神经网络解剖教室

桌面端神经网络教学演示系统（Tauri 2 + Rust + 原生 WebView）：手写数字识别的全程可视化。

定位：不是追求识别性能的工程系统，而是让学员亲眼看见神经网络在训练与推理时"内部发生了什么"。
从手写一个数字开始，到 169 个像素（13×13）进入 169→16→10 单隐藏层网络，
到 Softmax 输出 10 类概率，前向传导的粒子流动、反向传播的误差回流、
权重连线的粗细与颜色（蓝正红负）、混淆矩阵的逐格点亮，全程课堂可讲、可暂停、可回看。

## 快速开始

```bash
# 需要 Rust + Xcode Command Line Tools
cargo tauri dev     # 开发模式
cargo tauri build   # 打包 .app
```

数据保存在 ~/Documents/nn-anatomy-workspace/：
datasets/train|test/<label>/*.png 手写样本（真实 PNG 文件）+ models/*.nnmodel.json 模型。

内置 MNIST 精选数据集（1000 训练 / 300 测试），一键导入后 150 epoch 达 91%+ 测试准确率。

## 课堂流程

1. 采集：手写数字，点选标签，存入训练集
2. 前向：松手即看信号逐层传导、输出概率收敛
3. 训练：单步看权重更新 / 批量快进看 loss 下降
4. 泛化：用没见过的手写数字考它，看混淆矩阵

详细设计见 docs/design-spec.md 与 docs/implementation-plan.md。
