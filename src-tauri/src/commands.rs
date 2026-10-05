use crate::nn::network::{Arch, Network};
use crate::nn::optimizer::AdamState;
use crate::nn::preprocess::preprocess;
use crate::trainer::{evaluate, train_epoch, EvalReport, LabeledSample};
use crate::workspace::{ModelFile, SampleEntry, Source, Split, Workspace};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Shared app state: one workspace dir + the currently open model.
pub struct AppState {
    workspace: Workspace,
    model: ModelFile,
    adam: AdamState,
    epoch: usize,
    history: Vec<EpochPoint>,
}

#[derive(Clone, Serialize)]
pub struct EpochPoint {
    pub epoch: usize,
    pub loss: f64,
    pub acc: f64,
    pub test_acc: f64,
}

#[derive(Serialize)]
pub struct ModelSummary {
    pub name: String,
    pub epoch: usize,
    pub test_acc: f64,
}

#[derive(Serialize)]
pub struct DatasetStats {
    pub train_count: usize,
    pub test_count: usize,
}

#[derive(Serialize)]
pub struct SampleCard {
    pub id: String,
    pub label: u8,
    pub split: String,
    pub source: String,
    pub png_base64: String,
}

impl AppState {
    fn load_pixels(&self, entry: &SampleEntry) -> Result<Vec<f64>, String> {
        let bytes = self.workspace.read_sample_png(entry)?;
        let img = image::load_from_memory(&bytes).map_err(|e| format!("PNG 解码失败: {}", e))?;
        let gray = img.to_luma8();
        let (w, h) = gray.dimensions();
        let raw = gray.into_raw();
        Ok(preprocess(&raw, w as usize, h as usize))
    }

    fn labeled_samples(&self, split: Split) -> Result<Vec<LabeledSample>, String> {
        let entries = self.workspace.list_samples(split)?;
        let mut out = vec![];
        for e in entries {
            let x = self.load_pixels(&e)?;
            out.push(LabeledSample { x, y: e.label as usize });
        }
        Ok(out)
    }

    fn reload_workspace_into(&mut self, name: &str) -> Result<(), String> {
        self.model = self.workspace.load_model(name)?;
        let n = self.model.architecture.inputs * self.model.architecture.hidden
            + self.model.architecture.hidden
            + self.model.architecture.outputs * self.model.architecture.hidden
            + self.model.architecture.outputs;
        self.adam = AdamState::new(n);
        self.epoch = 0;
        self.history.clear();
        Ok(())
    }
}

#[tauri::command]
pub fn app_init() -> Result<ModelSummary, String> {
    let state = boot_state()?;
    *app_cell().lock().unwrap() = Some(state);
    current_summary()
}

const PARAMS: usize = 169 * 16 + 16 + 10 * 16 + 10;

// Tauri commands may run on different worker threads, so this must be a
// process-wide static rather than thread_local, or app_init's state would be
// lost for commands that land on another thread.
static APP: OnceLock<Mutex<Option<AppState>>> = OnceLock::new();

fn app_cell() -> &'static Mutex<Option<AppState>> {
    APP.get_or_init(|| Mutex::new(None))
}

fn with_state<T>(f: impl FnOnce(&mut AppState) -> Result<T, String>) -> Result<T, String> {
    let mut guard = app_cell()
        .lock()
        .map_err(|e| format!("状态锁中毒: {}", e))?;
    // Lazy auto-init: the frontend may issue reads before app_init lands
    // (multiple async modules boot in parallel), so initialize on first use.
    if guard.is_none() {
        let state = boot_state()?;
        *guard = Some(state);
    }
    f(guard.as_mut().unwrap())
}

fn boot_state() -> Result<AppState, String> {
    let dir = default_workspace_dir();
    let ws = Workspace::create(&dir)?;
    let names = ws.list_models()?;
    if names.is_empty() {
        let model = fresh_model("初始模型");
        ws.save_model(&model)?;
        Ok(AppState { workspace: ws, model, adam: AdamState::new(PARAMS), epoch: 0, history: vec![] })
    } else {
        let name = names[0].clone();
        let mut state = AppState {
            workspace: ws,
            model: fresh_model(&name),
            adam: AdamState::new(PARAMS),
            epoch: 0,
            history: vec![],
        };
        state.reload_workspace_into(&name)?;
        Ok(state)
    }
}

fn default_workspace_dir() -> PathBuf {
    let home = dirs_home().unwrap_or_else(|| PathBuf::from("."));
    home.join("Documents").join("nn-anatomy-workspace")
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}

fn fresh_model(name: &str) -> ModelFile {
    ModelFile {
        id: format!("m-{}", now_ms()),
        name: name.to_string(),
        architecture: Arch { inputs: 169, hidden: 16, outputs: 10 },
        weights: Network::new(Arch { inputs: 169, hidden: 16, outputs: 10 }, 999),
        optimizer_state: None,
        metrics: json!({}),
        trained_at: format!("{}", now_ms()),
    }
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

fn current_summary() -> Result<ModelSummary, String> {
    with_state(|s| {
        let test_acc = s.history.last().map(|h| h.test_acc).unwrap_or(0.0);
        Ok(ModelSummary { name: s.model.name.clone(), epoch: s.epoch, test_acc })
    })
}

// ===== dataset commands =====

#[tauri::command]
pub fn dataset_stats() -> Result<DatasetStats, String> {
    with_state(|s| {
        Ok(DatasetStats {
            train_count: s.workspace.list_samples(Split::Train)?.len(),
            test_count: s.workspace.list_samples(Split::Test)?.len(),
        })
    })
}

#[tauri::command]
pub fn dataset_list(split: String, offset: usize, limit: usize) -> Result<Vec<SampleCard>, String> {
    with_state(|s| {
        let sp = parse_split(&split)?;
        let entries = s.workspace.list_samples(sp)?;
        let mut out = vec![];
        for e in entries.iter().skip(offset).take(limit) {
            let bytes = s.workspace.read_sample_png(e)?;
            out.push(SampleCard {
                id: e.id.clone(),
                label: e.label,
                split: split.clone(),
                source: e.source.clone(),
                png_base64: base64_encode(&bytes),
            });
        }
        Ok(out)
    })
}

#[tauri::command]
pub fn dataset_add(
    split: String,
    label: u8,
    png_base64: String,
    width: u32,
    height: u32,
) -> Result<String, String> {
    with_state(|s| {
        let bytes = base64_decode(&png_base64)?;
        let sp = parse_split(&split)?;
        let source = if sp == Split::Train { Source::Handwritten } else { Source::Augmented };
        s.workspace.add_sample(sp, label, &bytes, width, height, source)
    })
}

#[tauri::command]
pub fn dataset_remove(split: String, id: String) -> Result<(), String> {
    with_state(|s| s.workspace.remove_sample(&id, parse_split(&split)?))
}

/// Imports the bundled 1000/300 MNIST-derived sample dataset into the
/// workspace as 13x13 grayscale PNGs. Returns (train, test) counts.
#[tauri::command]
pub fn dataset_import_builtin() -> Result<(usize, usize), String> {
    let ds = crate::sample_data::load()?;
    let (train_n, test_n) = with_state(|s| {
        let mut n = (0, 0);
        for (split, list) in [(Split::Train, &ds.train), (Split::Test, &ds.test)] {
            for sample in list {
                let png = gray13_to_png(&sample.pixels)?;
                s.workspace
                    .add_sample(split, sample.label, &png, 13, 13, Source::MnistImport)?;
                if split == Split::Train { n.0 += 1; } else { n.1 += 1; }
            }
        }
        Ok(n)
    })?;
    Ok((train_n, test_n))
}

// ===== model commands =====

#[tauri::command]
pub fn model_list() -> Result<Vec<String>, String> {
    with_state(|s| s.workspace.list_models())
}

#[tauri::command]
pub fn model_create(name: String, seed: Option<u64>) -> Result<ModelSummary, String> {
    with_state(|s| {
        if name.trim().is_empty() {
            return Err("模型名不能为空".into());
        }
        let seed = seed.unwrap_or(999);
        let model = ModelFile {
            id: format!("m-{}", now_ms()),
            name: name.trim().to_string(),
            architecture: Arch { inputs: 169, hidden: 16, outputs: 10 },
            weights: Network::new(Arch { inputs: 169, hidden: 16, outputs: 10 }, seed),
            optimizer_state: None,
            metrics: json!({}),
            trained_at: format!("{}", now_ms()),
        };
        s.workspace.save_model(&model)?;
        s.model = model;
        s.adam = AdamState::new(PARAMS);
        s.epoch = 0;
        s.history.clear();
        Ok(ModelSummary { name: s.model.name.clone(), epoch: 0, test_acc: 0.0 })
    })
}

#[tauri::command]
pub fn model_rename(new_name: String) -> Result<ModelSummary, String> {
    with_state(|s| {
        if new_name.trim().is_empty() {
            return Err("模型名不能为空".into());
        }
        let old = s.model.name.clone();
        s.model.name = new_name.trim().to_string();
        s.workspace.delete_model(&old)?;
        s.workspace.save_model(&s.model)?;
        Ok(ModelSummary { name: s.model.name.clone(), epoch: s.epoch, test_acc: 0.0 })
    })
}

#[tauri::command]
pub fn model_delete(name: String) -> Result<(), String> {
    with_state(|s| s.workspace.delete_model(&name))
}

#[tauri::command]
pub fn model_load(name: String) -> Result<ModelSummary, String> {
    with_state(|s| {
        s.reload_workspace_into(&name)?;
        let test_acc = s.model.metrics.get("test_acc").and_then(|v| v.as_f64()).unwrap_or(0.0);
        Ok(ModelSummary { name: s.model.name.clone(), epoch: s.epoch, test_acc })
    })
}

#[tauri::command]
pub fn model_export() -> Result<serde_json::Value, String> {
    with_state(|s| serde_json::to_value(&s.model).map_err(|e| e.to_string()))
}

#[derive(Deserialize)]
pub struct ImportPayload {
    pub name: String,
    pub weights: Network,
    pub epoch: Option<usize>,
}

#[tauri::command]
pub fn model_import(payload: ImportPayload) -> Result<ModelSummary, String> {
    with_state(|s| {
        let model = ModelFile {
            id: format!("m-{}", now_ms()),
            name: payload.name,
            architecture: payload.weights.arch.clone(),
            weights: payload.weights,
            optimizer_state: None,
            metrics: json!({}),
            trained_at: format!("{}", now_ms()),
        };
        s.workspace.save_model(&model)?;
        s.model = model;
        s.adam = AdamState::new(PARAMS);
        s.epoch = payload.epoch.unwrap_or(0);
        s.history.clear();
        Ok(ModelSummary { name: s.model.name.clone(), epoch: s.epoch, test_acc: 0.0 })
    })
}

// ===== training commands =====

#[tauri::command]
pub fn train_one_epoch(lr: f64, use_sgd: bool) -> Result<EpochPoint, String> {
    with_state(|s| {
        let samples = s.labeled_samples(Split::Train)?;
        if samples.is_empty() {
            return Err("训练集为空：请先手写或导入样本".into());
        }
        let report = train_epoch(&mut s.model.weights, &mut s.adam, &samples, lr, s.epoch, use_sgd);
        s.epoch += 1;
        let test = s.labeled_samples(Split::Test)?;
        let test_acc = if test.is_empty() { 0.0 } else { evaluate(&s.model.weights, &test).acc };
        let point = EpochPoint { epoch: s.epoch, loss: report.loss, acc: report.acc, test_acc };
        s.history.push(point.clone());
        persist(s)?;
        Ok(point)
    })
}

#[tauri::command]
pub fn train_bulk(epochs: usize, lr: f64, use_sgd: bool) -> Result<Vec<EpochPoint>, String> {
    with_state(|s| {
        let samples = s.labeled_samples(Split::Train)?;
        if samples.is_empty() {
            return Err("训练集为空：请先手写或导入样本".into());
        }
        let test = s.labeled_samples(Split::Test)?;
        let mut out = vec![];
        for _ in 0..epochs {
            let report = train_epoch(&mut s.model.weights, &mut s.adam, &samples, lr, s.epoch, use_sgd);
            s.epoch += 1;
            let test_acc = if test.is_empty() { 0.0 } else { evaluate(&s.model.weights, &test).acc };
            let point = EpochPoint { epoch: s.epoch, loss: report.loss, acc: report.acc, test_acc };
            s.history.push(point.clone());
            out.push(point);
        }
        persist(s)?;
        Ok(out)
    })
}

#[tauri::command]
pub fn training_history() -> Result<Vec<EpochPoint>, String> {
    with_state(|s| Ok(s.history.clone()))
}

#[tauri::command]
pub fn evaluate_test() -> Result<EvalReport, String> {
    with_state(|s| {
        let test = s.labeled_samples(Split::Test)?;
        if test.is_empty() {
            return Err("测试集为空".into());
        }
        let report = evaluate(&s.model.weights, &test);
        s.model.metrics = json!({ "test_acc": report.acc });
        persist(s)?;
        Ok(report)
    })
}

#[tauri::command]
pub fn training_reset() -> Result<ModelSummary, String> {
    with_state(|s| {
        s.adam = AdamState::new(PARAMS);
        s.epoch = 0;
        s.history.clear();
        persist(s)?;
        Ok(ModelSummary { name: s.model.name.clone(), epoch: 0, test_acc: 0.0 })
    })
}

fn persist(s: &mut AppState) -> Result<(), String> {
    s.model.trained_at = format!("{}", now_ms());
    s.model.metrics = json!({
        "epochs": s.epoch,
        "test_acc": s.history.last().map(|h| h.test_acc).unwrap_or(0.0),
    });
    s.workspace.save_model(&s.model)
}

// ===== helpers =====

fn parse_split(s: &str) -> Result<Split, String> {
    match s {
        "train" => Ok(Split::Train),
        "test" => Ok(Split::Test),
        _ => Err(format!("未知数据集: {}（应为 train 或 test）", s)),
    }
}

fn gray13_to_png(pixels: &[f64]) -> Result<Vec<u8>, String> {
    let img = image::GrayImage::from_fn(13, 13, |x, y| {
        let v = pixels[(y * 13 + x) as usize];
        image::Luma([(v * 255.0).round().clamp(0.0, 255.0) as u8])
    });
    let mut bytes = Vec::new();
    image::DynamicImage::ImageLuma8(img)
        .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
        .map_err(|e| format!("PNG 编码失败: {}", e))?;
    Ok(bytes)
}

fn base64_encode(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(TABLE[(n >> 18) as usize & 63] as char);
        out.push(TABLE[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { TABLE[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { TABLE[n as usize & 63] as char } else { '=' });
    }
    out
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let mut out = vec![];
    let bytes: Vec<u8> = s.bytes().filter(|b| *b != b'=' && !b.is_ascii_whitespace()).collect();
    for chunk in bytes.chunks(4) {
        if chunk.len() < 2 {
            return Err("非法 base64 长度".into());
        }
        let mut n: u32 = 0;
        for (i, &c) in chunk.iter().enumerate() {
            let d = table_pos(c)?;
            n |= (d as u32) << (18 - 6 * i);
        }
        out.push((n >> 16) as u8);
        // 2 chars -> 1 byte, 3-4 chars -> 2-3 bytes
        if chunk.len() == 2 { break; }
        out.push((n >> 8) as u8);
        if chunk.len() > 3 { out.push(n as u8); }
    }
    Ok(out)
}

fn table_pos(c: u8) -> Result<u8, String> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    TABLE.iter().position(|&t| t == c).map(|p| p as u8).ok_or_else(|| format!("非法 base64 字符: {}", c as char))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_roundtrip_all_lengths() {
        for len in 0..40 {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 + 11) as u8).collect();
            let enc = base64_encode(&data);
            // canonical encoder always maps back losslessly
            let dec = base64_decode(&enc).unwrap();
            assert_eq!(data, dec, "len {}", len);
        }
        // reject garbage inputs instead of silently truncating
        assert!(base64_decode("A").is_err());
    }

    #[test]
    fn gray13_png_roundtrip() {
        let mut pixels = vec![0.0f64; 169];
        pixels[40] = 0.9;
        pixels[41] = 0.5;
        let png = gray13_to_png(&pixels).unwrap();
        let img = image::load_from_memory(&png).unwrap().to_luma8();
        assert_eq!(img.dimensions(), (13, 13));
        let raw = img.into_raw();
        assert!((raw[40] as f64 - 229.5).abs() < 1.0, "v0.9 -> ~230, got {}", raw[40]);
        assert!((raw[41] as f64 - 127.5).abs() < 1.0, "v0.5 -> ~128, got {}", raw[41]);
    }

    #[test]
    fn train_one_epoch_via_app_state() {
        // Simulate the command layer flow on the bundled dataset: import
        // via workspace, train one epoch, expect finite loss and history.
        let dir = tempdir::TempDir::new("cmd").unwrap();
        let ws = Workspace::create(dir.path()).unwrap();
        let ds = crate::sample_data::load().unwrap();
        for s in ds.train.iter().take(50) {
            let png = gray13_to_png(&s.pixels).unwrap();
            ws.add_sample(Split::Train, s.label, &png, 13, 13, Source::MnistImport).unwrap();
        }
        for s in ds.test.iter().take(20) {
            let png = gray13_to_png(&s.pixels).unwrap();
            ws.add_sample(Split::Test, s.label, &png, 13, 13, Source::MnistImport).unwrap();
        }
        let model = fresh_model("cmd-test");
        let mut state = AppState {
            workspace: ws,
            model,
            adam: AdamState::new(PARAMS),
            epoch: 0,
            history: vec![],
        };
        let samples = state.labeled_samples(Split::Train).unwrap();
        assert_eq!(samples.len(), 50);
        let report = train_epoch(&mut state.model.weights, &mut state.adam, &samples, 0.001, 0, false);
        assert!(report.loss.is_finite() && report.loss > 0.0);
        state.epoch += 1;
        let test = state.labeled_samples(Split::Test).unwrap();
        let ev = evaluate(&state.model.weights, &test);
        assert!(ev.acc >= 0.0 && ev.acc <= 1.0);
    }
}
