#[cfg(test)]
mod tests {
    use super::*;

    fn sample_png(w: u32, h: u32) -> Vec<u8> {
        // A simple deterministic grayscale image: a filled square.
        let img = image::GrayImage::from_fn(w, h, |x, y| {
            if x > w / 4 && x < 3 * w / 4 && y > h / 4 && y < 3 * h / 4 {
                image::Luma([255u8])
            } else {
                image::Luma([0u8])
            }
        });
        let mut bytes = Vec::new();
        image::DynamicImage::ImageLuma8(img)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png)
            .expect("encode png");
        bytes
    }

    #[test]
    fn workspace_roundtrip_add_list_remove() {
        let dir = tempdir::TempDir::new("ws").unwrap();
        let ws = Workspace::create(dir.path()).unwrap();

        // Add a handwritten sample to train and one to test.
        let id1 = ws.add_sample(Split::Train, 7, &sample_png(260, 260), 260, 260, Source::Handwritten).unwrap();
        let id2 = ws.add_sample(Split::Test, 3, &sample_png(13, 13), 13, 13, Source::MnistImport).unwrap();
        assert_ne!(id1, id2);

        // PNG files land under the label directory.
        assert!(dir.path().join("datasets/train/7").join(format!("{}.png", id1)).exists());
        assert!(dir.path().join("datasets/test/3").join(format!("{}.png", id2)).exists());

        // Manifest round-trip: reload and list.
        let ws2 = Workspace::open(dir.path()).unwrap();
        let train = ws2.list_samples(Split::Train).unwrap();
        let test = ws2.list_samples(Split::Test).unwrap();
        assert_eq!(train.len(), 1);
        assert_eq!(test.len(), 1);
        assert_eq!(train[0].id, id1);
        assert_eq!(train[0].label, 7);
        assert_eq!(train[0].source, "handwritten");
        assert_eq!(test[0].source, "mnist-import");
        assert_eq!(train[0].width, 260);
        assert_eq!(test[0].width, 13);

        // Remove deletes both the manifest entry and the PNG.
        ws2.remove_sample(&id1, Split::Train).unwrap();
        assert!(!dir.path().join("datasets/train/7").join(format!("{}.png", id1)).exists());
        let train2 = ws2.list_samples(Split::Train).unwrap();
        assert!(train2.is_empty());
    }

    #[test]
    fn model_save_load_delete_roundtrip() {
        let dir = tempdir::TempDir::new("ws").unwrap();
        let ws = Workspace::create(dir.path()).unwrap();

        let model = ModelFile {
            id: "m1".to_string(),
            name: "课堂演示-第1版".to_string(),
            architecture: crate::nn::network::Arch { inputs: 256, hidden: 24, outputs: 10 },
            weights: crate::nn::network::Network {
                arch: crate::nn::network::Arch { inputs: 256, hidden: 24, outputs: 10 },
                w1: vec![0.5; 256 * 24],
                b1: vec![0.0; 24],
                w2: vec![0.25; 10 * 24],
                b2: vec![0.0; 10],
            },
            optimizer_state: Some(serde_json::json!({ "m": [0.1], "v": [0.01], "t": 3 })),
            metrics: serde_json::json!({ "epochsTrained": 3, "samplesSeen": 60 }),
            trained_at: "2026-10-04T23:00:00Z".to_string(),
        };
        ws.save_model(&model).unwrap();
        let path = dir.path().join("models/课堂演示-第1版.nnmodel.json");
        assert!(path.exists());

        let loaded = ws.load_model("课堂演示-第1版").unwrap();
        assert_eq!(loaded.id, model.id);
        assert_eq!(loaded.weights.w1.len(), 256 * 24);
        assert_eq!(loaded.weights.w2[0], 0.25);

        let names = ws.list_models().unwrap();
        assert_eq!(names, vec!["课堂演示-第1版".to_string()]);

        ws.delete_model("课堂演示-第1版").unwrap();
        assert!(!path.exists());
        assert!(ws.list_models().unwrap().is_empty());
    }

    #[test]
    fn create_is_idempotent_and_open_fails_on_missing() {
        let dir = tempdir::TempDir::new("ws").unwrap();
        let _ = Workspace::create(dir.path()).unwrap();
        // Creating twice must not fail (folders already exist).
        let _ = Workspace::create(dir.path()).unwrap();
        // Opening a non-workspace directory must fail.
        let dir2 = tempdir::TempDir::new("ws2").unwrap();
        assert!(Workspace::open(dir2.path()).is_err());
    }
}
use crate::nn::network::{Arch, Network};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Split {
    Train,
    Test,
}

impl Split {
    fn dir_name(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Test => "test",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Source {
    Handwritten,
    MnistImport,
    Augmented,
}

impl Source {
    fn as_str(self) -> &'static str {
        match self {
            Source::Handwritten => "handwritten",
            Source::MnistImport => "mnist-import",
            Source::Augmented => "augmented",
        }
    }

    fn from_str(s: &str) -> Source {
        match s {
            "mnist-import" => Source::MnistImport,
            "augmented" => Source::Augmented,
            _ => Source::Handwritten,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SampleEntry {
    pub id: String,
    pub label: u8,
    pub file: String,
    pub source: String,
    pub created_at: String,
    pub width: u32,
    pub height: u32,
}

/// One .nnmodel.json file under models/.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelFile {
    pub id: String,
    pub name: String,
    pub architecture: Arch,
    pub weights: Network,
    pub optimizer_state: Option<serde_json::Value>,
    pub metrics: serde_json::Value,
    pub trained_at: String,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    samples: Vec<SampleEntry>,
}

pub struct Workspace {
    root: PathBuf,
    id_counter: AtomicU64,
}

impl Workspace {
    /// Opens an existing workspace (must contain workspace.json).
    pub fn open(path: &Path) -> Result<Workspace, String> {
        let meta = path.join("workspace.json");
        if !meta.exists() {
            return Err(format!("不是有效的工作区（缺少 workspace.json）: {}", path.display()));
        }
        Ok(Workspace {
            root: path.to_path_buf(),
            id_counter: AtomicU64::new(0),
        })
    }

    /// Creates (or re-opens) a workspace directory layout.
    pub fn create(path: &Path) -> Result<Workspace, String> {
        for split in ["train", "test"] {
            fs::create_dir_all(path.join("datasets").join(split))
                .map_err(|e| format!("创建目录失败: {}", e))?;
            for label in 0..10u8 {
                fs::create_dir_all(path.join("datasets").join(split).join(label.to_string()))
                    .map_err(|e| format!("创建目录失败: {}", e))?;
            }
        }
        fs::create_dir_all(path.join("models")).map_err(|e| format!("创建目录失败: {}", e))?;
        fs::create_dir_all(path.join("exports")).map_err(|e| format!("创建目录失败: {}", e))?;
        let meta = path.join("workspace.json");
        if !meta.exists() {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_secs();
            let content = serde_json::json!({ "version": 1, "createdAt": now });
            fs::write(&meta, serde_json::to_string_pretty(&content).unwrap())
                .map_err(|e| format!("写入 workspace.json 失败: {}", e))?;
        }
        Ok(Workspace {
            root: path.to_path_buf(),
            id_counter: AtomicU64::new(0),
        })
    }

    fn manifest_path(&self, split: Split) -> PathBuf {
        self.root.join("datasets").join(split.dir_name()).join("manifest.json")
    }

    fn load_manifest(&self, split: Split) -> Manifest {
        let path = self.manifest_path(split);
        match fs::read_to_string(&path) {
            Ok(text) => serde_json::from_str(&text).unwrap_or(Manifest { samples: vec![] }),
            Err(_) => Manifest { samples: vec![] },
        }
    }

    fn save_manifest(&self, split: Split, manifest: &Manifest) -> Result<(), String> {
        let path = self.manifest_path(split);
        let text = serde_json::to_string_pretty(manifest).map_err(|e| e.to_string())?;
        fs::write(&path, text).map_err(|e| format!("写入 manifest 失败: {}", e))
    }

    /// Lists all samples in a split, sorted by creation time.
    pub fn list_samples(&self, split: Split) -> Result<Vec<SampleEntry>, String> {
        let mut manifest = self.load_manifest(split);
        manifest.samples.sort_by(|a, b| a.created_at.cmp(&b.created_at));
        Ok(manifest.samples)
    }

    /// Adds a sample: writes the PNG under datasets/<split>/<label>/, then
    /// records it in the manifest. Returns the generated id.
    pub fn add_sample(
        &self,
        split: Split,
        label: u8,
        png_bytes: &[u8],
        width: u32,
        height: u32,
        source: Source,
    ) -> Result<String, String> {
        if label > 9 {
            return Err(format!("标签越界: {}（应为 0-9）", label));
        }
        let id = self.next_id();
        let file_name = format!("{}.png", id);
        let dir = self.root.join("datasets").join(split.dir_name()).join(label.to_string());
        let file = dir.join(&file_name);
        fs::write(&file, png_bytes).map_err(|e| format!("写入 PNG 失败: {}", e))?;

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis()
            .to_string();
        let entry = SampleEntry {
            id: id.clone(),
            label,
            file: format!("datasets/{}/{}/{}", split.dir_name(), label, file_name),
            source: source.as_str().to_string(),
            created_at,
            width,
            height,
        };
        let mut manifest = self.load_manifest(split);
        manifest.samples.push(entry);
        self.save_manifest(split, &manifest)?;
        Ok(id)
    }

    /// Adds a sample with a caller-provided id (used for idempotent builtin
    /// imports). Fails if the id already exists, mirroring add_sample's
    /// "no duplicate" guarantee. Returns the id on success.
    pub fn add_sample_with_id(
        &self,
        split: Split,
        label: u8,
        png_bytes: &[u8],
        width: u32,
        height: u32,
        source: Source,
        id: &str,
    ) -> Result<String, String> {
        if label > 9 {
            return Err(format!("标签越界: {}（应为 0-9）", label));
        }
        let file_name = format!("{}.png", id);
        let dir = self.root.join("datasets").join(split.dir_name()).join(label.to_string());
        let file = dir.join(&file_name);
        fs::write(&file, png_bytes).map_err(|e| format!("写入 PNG 失败: {}", e))?;

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| e.to_string())?
            .as_millis()
            .to_string();
        let entry = SampleEntry {
            id: id.to_string(),
            label,
            file: format!("datasets/{}/{}/{}", split.dir_name(), label, file_name),
            source: source.as_str().to_string(),
            created_at,
            width,
            height,
        };
        let mut manifest = self.load_manifest(split);
        if manifest.samples.iter().any(|s| s.id == id) {
            return Err(format!("样本已存在: {}", id));
        }
        manifest.samples.push(entry);
        self.save_manifest(split, &manifest)?;
        Ok(id.to_string())
    }

    /// Removes a sample: deletes the PNG and the manifest entry.
    pub fn remove_sample(&self, id: &str, split: Split) -> Result<(), String> {
        let mut manifest = self.load_manifest(split);
        let removed: Vec<SampleEntry> = manifest.samples.iter().filter(|s| s.id == id).cloned().collect();
        manifest.samples.retain(|s| s.id != id);
        if removed.is_empty() {
            return Err(format!("样本不存在: {}", id));
        }
        for s in &removed {
            let path = self.root.join(&s.file);
            let _ = fs::remove_file(&path);
        }
        self.save_manifest(split, &manifest)?;
        Ok(())
    }

    /// Resets the auto-increment id counter (used after dataset_clear_all so
    /// handwritten ids stay small and readable in class).
    pub fn reset_id_counter(&self) -> Result<(), String> {
        self.id_counter.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// timestamp-based id with an in-process counter to avoid collisions.
    fn next_id(&self) -> String {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        let n = self.id_counter.fetch_add(1, Ordering::SeqCst);
        format!("s{}-{}", now, n)
    }

    /// Saves a model as models/<name>.nnmodel.json (text, human-inspectable).
    pub fn save_model(&self, model: &ModelFile) -> Result<(), String> {
        let path = self.model_path(&model.name);
        let text = serde_json::to_string_pretty(model).map_err(|e| e.to_string())?;
        fs::write(&path, text).map_err(|e| format!("保存模型失败: {}", e))
    }

    pub fn load_model(&self, name: &str) -> Result<ModelFile, String> {
        let path = self.model_path(name);
        let text = fs::read_to_string(&path).map_err(|_| format!("模型不存在: {}", name))?;
        serde_json::from_str(&text).map_err(|e| format!("模型解析失败: {}", e))
    }

    pub fn list_models(&self) -> Result<Vec<String>, String> {
        let dir = self.root.join("models");
        let mut names = vec![];
        let entries = fs::read_dir(&dir).map_err(|e| format!("读取 models 失败: {}", e))?;
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".nnmodel.json") {
                names.push(name.trim_end_matches(".nnmodel.json").to_string());
            }
        }
        names.sort();
        Ok(names)
    }

    pub fn delete_model(&self, name: &str) -> Result<(), String> {
        let path = self.model_path(name);
        fs::remove_file(&path).map_err(|e| format!("删除模型失败: {}", e))
    }

    fn model_path(&self, name: &str) -> PathBuf {
        self.root.join("models").join(format!("{}.nnmodel.json", name))
    }

    /// Reads the raw PNG bytes of a sample (used by training/inference).
    pub fn read_sample_png(&self, entry: &SampleEntry) -> Result<Vec<u8>, String> {
        fs::read(self.root.join(&entry.file)).map_err(|e| format!("读取样本失败: {}", e))
    }

    /// Moves imported MNIST idx files into the workspace (Task 7 uses this).
    pub fn import_sample_dataset(&self) -> Result<usize, String> {
        Err("尚未实现：等待 Task 7 的 MNIST idx 导入器".to_string())
    }
}
