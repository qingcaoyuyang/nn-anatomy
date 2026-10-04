Task 0: docs committed at 01fc3ab (spec approved, plan approved)
Task 1: complete (commits 01fc3ab..06fbee0, review clean/approved)
  Minor findings (for final review): fonts not yet bundled (deferred to UI task); src/tauri-bridge.js unguarded window.__TAURI__; image crate default features pulls AVIF stack (trim when used); task-1-report cites stale hash b1494d2 vs actual 06fbee0
## Task 2: Rust 网络核心

- **状态**: ✅ 完成
- **实现**: 提交 0b1f4e8 + 修复 8a06fd0（lib target 拆分解决编译瓶颈，cargo test --lib 0.5s）
- **评审**: Darwin - 首轮 Needs fixes（2 Important）→ 修复后复审 Approved
- **修复**: 规范 mulberry32 + fold 种子混合 + 输入断言 + z2/g_b2 测试覆盖 + RNG 黄金值跨语言奇偶测试
- **测试**: 6 passed / 0 failed, 0 warnings
