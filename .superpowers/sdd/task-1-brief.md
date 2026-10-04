### Task 1: Tauri 2 项目骨架

**Files:**
- Create: src-tauri/Cargo.toml
- Create: src-tauri/tauri.conf.json
- Create: src-tauri/src/main.rs
- Create: src-tauri/build.rs
- Create: src-tauri/icons/（cargo tauri icon 生成）
- Create: index.html
- Create: src/main.js
- Create: src/tauri-bridge.js
- Create: src/ui/app.css
- Create: .gitignore

**Interfaces:**
- Produces: 可运行的 cargo tauri dev 骨架，后续任务在其上叠加
- Produces: Rust command `greet(name: String) -> String`（占位，验证前后端桥接）

- [ ] **Step 1: 初始化**

```bash
cd /Users/juewei/Documents/Codex/OtherWork/nn-anatomy
cargo install tauri-cli --locked
mkdir -p src-tauri/src src/ui src/nn tests scripts
touch index.html src/main.js src/tauri-bridge.js src/ui/app.css src-tauri/src/main.rs src-tauri/build.rs src-tauri/Cargo.toml src-tauri/tauri.conf.json .gitignore
```

- [ ] **Step 2: 写配置与骨架代码**

```toml
# src-tauri/Cargo.toml
[package]
name = "nn-anatomy"
version = "0.1.0"
edition = "2021"

[build-dependencies]
tauri-build = { version = "2", features = [] }

[dependencies]
tauri = { version = "2", features = [] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
image = "0.25"
tempdir = { version = "0.3", optional = true }

[dev-dependencies]
tempdir = "0.3"
```

```rust
// src-tauri/src/main.rs
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[tauri::command]
fn greet(name: &str) -> String {
    format!("你好，{}！神经网络解剖教室准备就绪。", name)
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

```json
// src-tauri/tauri.conf.json
{
  "productName": "nn-anatomy",
  "version": "0.1.0",
  "identifier": "com.juewei.nn-anatomy",
  "build": { "frontendDist": "../", "devUrl": "http://localhost:5173" },
  "app": {
    "withGlobalTauri": true,
    "windows": [{ "title": "nn-anatomy 神经网络解剖教室", "width": 1280, "height": 800 }]
  },
  "bundle": { "active": true, "targets": ["dmg", "app"] }
}
```

```rust
// src-tauri/build.rs
fn main() { tauri_build::build() }
```

```html
<!-- index.html -->
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>nn-anatomy 神经网络解剖教室</title>
  <link rel="stylesheet" href="src/ui/app.css" />
</head>
<body>
  <div id="app"></div>
  <script type="module" src="src/main.js"></script>
</body>
</html>
```

```js
// src/main.js
import { greet } from './tauri-bridge.js';
document.querySelector('#app').textContent = await greet('舟哥');
```

```js
// src/tauri-bridge.js
export async function invoke(cmd, args) {
  return window.__TAURI__.core.invoke(cmd, args);
}
export async function greet(name) { return invoke('greet', { name }); }
```

- [ ] **Step 3: 首次编译验证**

Run: cd /Users/juewei/Documents/Codex/OtherWork/nn-anatomy && cargo tauri dev
Expected: 窗口打开，显示中文问候

- [ ] **Step 4: 应用图标（教学视觉一致性）**

```bash
# 用 Canvas 画 1024×1024 节点连线图标导出 PNG 后：
cargo tauri icon icon-1024.png
```

- [ ] **Step 5: 提交 git commit -m "feat: tauri skeleton with bridge"**

