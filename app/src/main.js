import { initDrawing } from './ui/drawing.js';
import { initStage } from './ui/stage.js';
import { createNetwork, forward } from './nn/network.js';
import { initPanel } from './ui/panel.js';
import { initDatasetViewer } from './ui/dataset-viewer.js';
import { invoke } from './tauri-bridge.js';

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function renderPreview(pixels13, previewCanvas) {
  const ctx = previewCanvas.getContext('2d');
  const cell = previewCanvas.width / 13;
  ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  for (let y = 0; y < 13; y++) {
    for (let x = 0; x < 13; x++) {
      const v = pixels13[y * 13 + x];
      ctx.fillStyle = 'rgba(249,115,22,' + v.toFixed(3) + ')';
      ctx.fillRect(x * cell, y * cell, cell, cell);
    }
  }
  ctx.strokeStyle = 'rgba(55,65,81,0.15)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 13; i++) {
    ctx.beginPath();
    ctx.moveTo(i * cell, 0);
    ctx.lineTo(i * cell, previewCanvas.height);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i * cell);
    ctx.lineTo(previewCanvas.width, i * cell);
    ctx.stroke();
  }
}

// Convert the backend ModelFile JSON (w1/b1/w2/b2 lowercase) to the JS
// engine's net shape (W1/b1/W2/b2 uppercase) so the stage renders the REAL
// trained weights, not a stale JS-side copy.
function backendNetToJs(m) {
  const w = m.weights;
  return {
    arch: { inputs: w.arch.inputs, hidden: w.arch.hidden, outputs: w.arch.outputs },
    W1: w.w1, b1: w.b1, W2: w.w2, b2: w.b2,
  };
}

// ============ View shell: header + three switchable views ============

const app = document.getElementById('app');
app.innerHTML = '';
const shell = el('div', 'app-shell');

const header = el('header', 'app-header');
const titleBox = el('div', 'title-box');
titleBox.append(el('span', 'logo-dot'));
titleBox.append(el('h1', null, '神经网络解剖教室'));
titleBox.append(el('span', 'title-sub', '169 → 16 → 10 · 看见每一次权重更新'));
header.append(titleBox);

const stepBadge = el('div', 'step-badge');
stepBadge.append(el('span', 'step-badge-num', '第 1 步'));
stepBadge.append(el('span', 'step-badge-hint', '共 3 步 · 采集数据'));
header.append(stepBadge);

const tabs = el('nav', 'stage-tabs');
const tabDefs = [
  { key: 'collect', name: '① 采集数据', sub: '手写并标注' },
  { key: 'train', name: '② 训练模型', sub: '调整权重' },
  { key: 'test', name: '③ 测试识别', sub: '验证效果' },
];
const tabBtns = {};
for (const t of tabDefs) {
  const b = el('button', 'stage-tab');
  b.append(el('span', 'stage-tab-name', t.name));
  b.append(el('span', 'stage-tab-sub', t.sub));
  b.addEventListener('click', () => showView(t.key));
  tabs.append(b);
  tabBtns[t.key] = b;
}
header.append(tabs);
shell.append(header);

// ============ Shared live model state ============

const ARCH = { inputs: 169, hidden: 16, outputs: 10 };
let liveNet = createNetwork(ARCH, 999); // replaced by real backend weights on boot
let lastInference = null; // { p: [10], pixels }
let currentView = 'collect';

const views = {};

function showView(key) {
  currentView = key;
  for (const k of Object.keys(views)) views[k].classList.toggle('view-hidden', k !== key);
  for (const k of Object.keys(tabBtns)) tabBtns[k].classList.toggle('active', k === key);
  const idx = tabDefs.findIndex((t) => t.key === key);
  stepBadge.querySelector('.step-badge-num').textContent = '第 ' + (idx + 1) + ' 步';
  stepBadge.querySelector('.step-badge-hint').textContent = '共 3 步 · ' + tabDefs[idx].name.replace(/^[①②③]s*/, '');
  if (key === 'train' && trainStage) trainStage.drawBase();
  if (key === 'test' && testStage) testStage.drawBase();
}

// ============ View 1: collect (drawing + save + dataset) ============

const collectView = el('main', 'view view-collect');
views.collect = collectView;

const collectHint = el('div', 'view-hero');
collectHint.append(el('h2', null, '第一步：制作你的数据集'));
collectHint.append(el('p', null, '写一个数字 → 点选它对应的标签 → 保存到训练集。每个数字写 5-10 个不同样式，模型才能学到共性。'));
collectView.append(collectHint);

const collectCols = el('div', 'collect-cols');
const collectLeft = el('section', 'panel');
const padWrap = el('div', 'pad-wrap');
const canvas = el('canvas', 'draw-pad');
canvas.width = 260;
canvas.height = 260;
padWrap.append(canvas);
const clearBtn = el('button', 'btn btn-ghost', '清空');
padWrap.append(clearBtn);
collectLeft.append(padWrap);

const previewBox = el('div', 'grid-box');
const preview = el('canvas', 'grid-preview');
preview.width = 169;
preview.height = 169;
previewBox.append(preview);
collectLeft.append(previewBox);

const labelTitle = el('p', 'label-title', '这个数字是几？点选标签');
const labelRow = el('div', 'label-row');
const labelBtns = [];
for (let d = 0; d <= 9; d++) {
  const b = el('button', 'digit-btn', String(d));
  b.addEventListener('click', () => {
    labelBtns.forEach((x) => x.classList.toggle('selected', x === b));
  });
  labelBtns.push(b);
  labelRow.append(b);
}
collectLeft.append(labelTitle, labelRow);

const saveRow = el('div', 'save-row');
const toTrain = el('button', 'btn btn-primary', '保存到训练集');
const toTest = el('button', 'btn btn-ghost', '保存到测试集');
saveRow.append(toTrain, toTest);
collectLeft.append(saveRow);
const saveNote = el('p', 'save-note', '保存后自动清空画板，可继续写下一个');
collectLeft.append(saveNote);

const nextToTrain = el('button', 'btn btn-next', '数据够了，去训练 →');
collectLeft.append(nextToTrain);
collectCols.append(collectLeft);

const collectRight = el('section', 'panel');
collectRight.append(el('h2', 'panel-title', '已采集的样本'));
const datasetRoot = el('div', 'dataset-root');
collectRight.append(datasetRoot);
collectCols.append(collectRight);
collectView.append(collectCols);
shell.append(collectView);

// ============ View 2: train (network + controls + history) ============

const trainView = el('main', 'view view-train');
views.train = trainView;

const trainHero = el('div', 'view-hero');
trainHero.append(el('h2', null, '第二步：训练模型'));
trainHero.append(el('p', null, '「训练 1 轮」= 模型看一遍全部样本并微调一次权重。新手建议直接点「连续训练 30 轮」；识别率还不稳就再点几次。观察连线粗细与热力图变化——那就是权重在被"拧紧"。'));
trainView.append(trainHero);

const trainCols = el('div', 'train-cols');
const trainStagePanel = el('section', 'panel');
const stageCanvas = el('canvas', 'stage-canvas');
stageCanvas.width = 900;
stageCanvas.height = 560;
trainStagePanel.append(stageCanvas);
trainCols.append(trainStagePanel);

const trainPanelRoot = el('div', 'train-controls-root');
trainCols.append(trainPanelRoot);
trainView.append(trainCols);
shell.append(trainView);

const goTestBtn = el('button', 'btn btn-next', '训练好了，去测试 →');
trainView.append(goTestBtn);

// ============ View 3: test (draw + infer + result) ============

const testView = el('main', 'view view-test');
views.test = testView;

const testHero = el('div', 'view-hero');
testHero.append(el('h2', null, '第三步：测试模型'));
testHero.append(el('p', null, '写一个模型没见过的数字（或从测试集取一张），看它识别成几。概率条最长的就是模型的答案——它反映的是当前权重对这个手写形状的"置信度"。'));
testView.append(testHero);

const testCols = el('div', 'test-cols');
const testLeft = el('section', 'panel');
const testPadWrap = el('div', 'pad-wrap');
const testCanvas = el('canvas', 'draw-pad');
testCanvas.width = 260;
testCanvas.height = 260;
testPadWrap.append(testCanvas);
const testClearBtn = el('button', 'btn btn-ghost', '清空');
testPadWrap.append(testClearBtn);
testLeft.append(testPadWrap);

const inferBtn = el('button', 'btn btn-primary btn-infer', '识别这个数字 →');
testLeft.append(inferBtn);
const testNote = el('p', 'save-note', '模型会输出 0-9 每个数字的概率');
testLeft.append(testNote);
testCols.append(testLeft);

const testRight = el('section', 'panel');
const testStageCanvas = el('canvas', 'stage-canvas stage-canvas-small');
testStageCanvas.width = 900;
testStageCanvas.height = 420;
testRight.append(testStageCanvas);
const probBox = el('div', 'prob-box');
testRight.append(probBox);
testCols.append(testRight);
testView.append(testCols);
shell.append(testView);

app.append(shell);

// ============ Wire up ============

const drawing = initDrawing(canvas, {
  lineWidth: 16,
  onStroke: (pixels13) => renderPreview(pixels13, preview),
});
const testDrawing = initDrawing(testCanvas, {
  lineWidth: 16,
  onStroke: (pixels13, strokes) => {
    if (strokes > 0 && testStage) testStage.playInference(pixels13);
  },
});

const trainStage = initStage(stageCanvas, liveNet);
const testStage = initStage(testStageCanvas, liveNet);
const panel = initPanel(trainPanelRoot, {
  onEpoch: (point) => {
    syncWeights().then(() => {
      trainStage.drawBase();
      testStage.drawBase();
    });
    if (point && point.test_acc >= 0.9) markDone('test90');
  },
  onError: (msg) => notify(msg, true),
  onModelChange: () => {
    syncWeights().then(() => {
      trainStage.drawBase();
      testStage.drawBase();
    });
  },
});
const viewer = initDatasetViewer(datasetRoot, {
  onChange: (stats) => {
    saveNote.textContent = '训练集 ' + stats.train_count + ' 张 / 测试集 ' + stats.test_count + ' 张';
    saveNote.classList.remove('save-note-error');
  },
});

// sync the JS-side network with the backend's current model weights
async function syncWeights() {
  const m = await invoke('model_export');
  liveNet = backendNetToJs(m);
  trainStage.setNetwork(liveNet);
  testStage.setNetwork(liveNet);
}

function notify(msg, isError, target) {
  const note = target || saveNote;
  note.textContent = msg;
  note.classList.toggle('save-note-error', Boolean(isError));
}

function markDone(key) {
  stepBadge.querySelector('.step-badge-hint').textContent = '✓ 测试准确率已超过 90%';
}

function drawProbBars(p) {
  probBox.innerHTML = '';
  const title = el('p', 'prob-title', '各数字的识别概率');
  probBox.append(title);
  const max = Math.max(...p);
  for (let d = 0; d < 10; d++) {
    const row = el('div', 'prob-row');
    const label = el('span', 'prob-label', String(d));
    const barWrap = el('div', 'prob-bar-wrap');
    const bar = el('div', 'prob-bar');
    bar.style.width = (p[d] * 100).toFixed(1) + '%';
    if (p[d] === max) bar.classList.add('prob-bar-top');
    barWrap.append(bar);
    const val = el('span', 'prob-val', (p[d] * 100).toFixed(1) + '%');
    row.append(label, barWrap, val);
    probBox.append(row);
  }
}

async function doInference() {
  if (testDrawing.isBlank()) { notify('请先写一个数字', true, testNote); return; }
  try {
    await syncWeights();
    const pixels = testDrawing.getPixels13();
    const f = await testStage.playInference(pixels);
    drawProbBars(f.p);
    const top = f.p.indexOf(Math.max(...f.p));
    notify('模型认为是 ' + top + '（置信度 ' + (f.p[top] * 100).toFixed(1) + '%）', false, testNote);
  } catch (e) {
    notify('推理失败: ' + (e.message || String(e)), true, testNote);
  }
}
inferBtn.addEventListener('click', doInference);

toTrain.addEventListener('click', async () => {
  if (drawing.isBlank()) { notify('请先写一个数字', true); return; }
  const selected = document.querySelector('.digit-btn.selected');
  if (!selected) { notify('请先点选这个数字的标签（0-9）', true); return; }
  const label = Number(selected.textContent);
  try {
    const dataUrl = canvas.toDataURL('image/png');
    const b64 = dataUrl.split(',')[1];
    await invoke('dataset_add', { split: 'train', label, pngBase64: b64, width: 260, height: 260 });
    drawing.clear();
    renderPreview(new Array(169).fill(0), preview);
    notify('已保存到训练集', false);
    viewer.refresh(true);
  } catch (e) {
    notify('保存失败: ' + (e.message || String(e)), true);
  }
});

toTest.addEventListener('click', async () => {
  if (drawing.isBlank()) { notify('请先写一个数字', true); return; }
  try {
    const dataUrl = canvas.toDataURL('image/png');
    const b64 = dataUrl.split(',')[1];
    await invoke('dataset_add', { split: 'test', label: 0, pngBase64: b64, width: 260, height: 260 });
    drawing.clear();
    renderPreview(new Array(169).fill(0), preview);
    notify('已保存到测试集（之后可在测试页考察它）', false);
    viewer.refresh(true);
  } catch (e) {
    notify('保存失败: ' + (e.message || String(e)), true);
  }
});

clearBtn.addEventListener('click', () => {
  drawing.clear();
  renderPreview(new Array(169).fill(0), preview);
});
testClearBtn.addEventListener('click', () => {
  testDrawing.clear();
  if (testStage) testStage.drawBase();
});

nextToTrain.addEventListener('click', () => showView('train'));
goTestBtn.addEventListener('click', () => showView('test'));

// click a hidden node on the train stage to inspect its weight heatmap
stageCanvas.addEventListener('click', (e) => {
  const rect = stageCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (stageCanvas.width / rect.width);
  const y = (e.clientY - rect.top) * (stageCanvas.height / rect.height);
  let hit = -1;
  for (let h = 0; h < 16; h++) {
    const ny = stageCanvas.height * ((h + 1) / 17);
    if (Math.abs(y - ny) < 14 && Math.abs(x - stageCanvas.width * 0.6) < 20) hit = h;
  }
  trainStage.highlightNode(1, hit);
});

// Boot: init backend then load the real model weights into both stages
panel.init()
  .then(() => syncWeights())
  .then(() => { trainStage.drawBase(); testStage.drawBase(); })
  .catch((e) => notify('初始化失败: ' + (e.message || String(e)), true));

showView('collect');
