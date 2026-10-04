export async function invoke(cmd, args) {
  if (!window.__TAURI__) {
    throw new Error('此功能需要在 Tauri 桌面应用中运行（当前是浏览器预览模式）');
  }
  return window.__TAURI__.core.invoke(cmd, args);
}

export async function greet(name) {
  return invoke('greet', { name });
}
