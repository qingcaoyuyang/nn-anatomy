# Task 1 报告：Tauri 2 项目骨架

## 状态
DONE_WITH_CONCERNS（功能达成，有一处偏离 brief 的必要修正）

## 实现内容
- Tauri 2 骨架：src-tauri/（Cargo.toml、build.rs、src/main.rs、tauri.conf.json、icons/icon.png）
- 前端零依赖：index.html、src/main.js、src/tauri-bridge.js、src/ui/app.css；预留 src/nn/、tests/、scripts/
- Rust command greet(name: &str) -> String 返回中文问候，前端经 window.__TAURI__.core.invoke 调用

## 歧义裁决落实
1. tauri.conf.json 删除 devUrl/beforeDevCommand，仅保留 frontendDist: "../"
2. 桥接走 window.__TAURI__.core.invoke（withGlobalTauri: true）
3. bundle.active: false，跳过 cargo tauri icon
4. 图标用 32x32 橙色 PNG 占位（见偏离说明）

## 偏离与修正
- brief 文件清单未含 src-tauri/icons/icon.png，但 generate_context! 宏编译期硬性要求该文件。首次 cargo build 报错 failed to open icon。补入 32x32 不透明 PNG，属可编译的最小必要文件。
- 第一次占位图标数据损坏导致运行时 invalid icon panic；重新生成有效 PNG 后 cargo tauri dev 无 panic。

## 验证结果
1. cargo install tauri-cli --locked 成功，tauri-cli v2.12.1（约 2m37s）
2. src-tauri 下 cargo build 成功，产出 target/debug/nn-anatomy
3. cargo tauri dev：Running target/debug/nn-anatomy 出现，无 panic 无错误，进程存活至手动终止，符合裁决 4 的通过标准（自动 CLI 环境无 GUI 断言）

## 提交
- b1494d2 feat: tauri skeleton with bridge

## 遗留
- 图标为纯色占位，后续 UI 任务需 cargo tauri icon 精修
- dev 验证未做 GUI 截图断言，后续接入 UI 后可补
- image/tempdir 依赖暂未引用，按 brief 保留
