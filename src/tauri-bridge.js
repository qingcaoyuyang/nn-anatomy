export async function invoke(cmd, args) {
  return window.__TAURI__.core.invoke(cmd, args);
}

export async function greet(name) {
  return invoke('greet', { name });
}
