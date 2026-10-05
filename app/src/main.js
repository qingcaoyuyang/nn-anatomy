import { initDrawing } from './ui/drawing.js';
import { initStage } from './ui/stage.js';
import { initLesson } from './ui/lesson.js';
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

// ============ App shell: topbar + lesson bar + three columns ============

const app = document.getElementById('app');
app.innerHTML = '';
const shell = el('div', 'app-shell');

const topbar = el('header', 'topbar');
const titleBox = el('div', 'title-box');
titleBox.append(el('span', 'logo-dot'));
titleBox.append(el('h1', null, '神经网络解剖教室'));
titleBox.append(el('span', 'title-sub', '256 → 24 → 10 · 看见每一次权重更新'));
topbar.append(titleBox);
shell.append(topbar);

const lessonRoot = el('div', 'lesson-root');
shell.append(lessonRoot);
const lesson = initLesson(lessonRoot);

const layout = el('main', 'layout');
const leftPanel = el('section', 'card panel-left');
const centerPanel = el('section', 'card panel-center');
const rightPanel = el('aside', 'card panel-right');
layout.append(leftPanel, centerPanel, rightPanel);
shell.append(layout);
app.append(shell);

// ============ Left column: draw + label + save ============

leftPanel.append(el('h2', 'card-title', '① 数据采集'));
const padWrap = el('div', 'pad-wrap');
const canvas = el('canvas', 'draw-pad');
canvas.width = 260;
canvas.height = 260;
padWrap.append(canvas);
const clearBtn = el('button', 'btn btn-ghost btn-sm', '清空');
padWrap.append(clearBtn);
leftPanel.append(padWrap);

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
leftPanel.append(labelTitle, labelRow);

const saveRow = el('div', 'save-row');
const toTrain = el('button', 'btn btn-primary', '保存到训练集');
const toTest = el('button', 'btn btn-ghost', '保存到测试集');
saveRow.append(toTrain, toTest);
leftPanel.append(saveRow);
const saveNote = el('p', 'save-note', '写一个数字 → 选标签 → 保存。每个数字多写几个不同样式。');
leftPanel.append(saveNote);

const datasetBlock = el('div', 'dataset-block');
const datasetSummary = el('p', 'ds-summary', '训练 0 · 测试 0');
datasetBlock.append(datasetSummary);
const datasetRoot = el('div', 'dataset-root');
datasetBlock.append(datasetRoot);
leftPanel.append(datasetBlock);

// ============ Center column: network stage + inference ============

centerPanel.append(el('h2', 'card-title', '② 神经网络舞台'));
const stageCanvas = el('canvas', 'stage-canvas');
centerPanel.append(stageCanvas);

const stageActions = el('div', 'stage-actions');
const inferBtn = el('button', 'btn btn-primary', '▶ 识别这个数字');
stageActions.append(inferBtn);
const inferHint = el('span', 'infer-hint', '把当前画板内容做一次前向传播');
stageActions.append(inferHint);
centerPanel.append(stageActions);
const inferNote = el('p', 'save-note', '写一个数字，点「识别」，看信号逐层流向答案。');
centerPanel.append(inferNote);

const probBox = el('div', 'prob-box');
centerPanel.append(probBox);

// ============ Right column: training panel ============

rightPanel.append(el('h2', 'card-title', '③ 训练与评估'));
const panelRoot = el('div', 'train-controls-root');
rightPanel.append(panelRoot);

// ============ Shared live model state ============

const ARCH = { inputs: 256, hidden: 24, outputs: 10 };
let liveNet = createNetwork(ARCH, 999); // replaced by real backend weights on boot
const stage = initStage(stageCanvas, liveNet);

const panel = initPanel(panelRoot, {
  onEpoch: () => {
    syncWeights().then(() => {
      stage.render();
    });
    lesson.notify('trained');
  },
  onError: (msg) => notify(msg, true, inferNote),
  onModelChange: () => {
    syncWeights().then(() => {
      stage.render();
    });
  },
  onEvaluate: () => lesson.notify('evaluated'),
});
const viewer = initDatasetViewer(datasetRoot, {
  onChange: (stats) => {
    datasetSummary.textContent = '训练 ' + stats.train_count + ' · 测试 ' + stats.test_count + ' 张';
  },
});

// sync the JS-side network with the backend's current model weights
async function syncWeights() {
  const m = await invoke('model_export');
  liveNet = backendNetToJs(m);
  stage.setNetwork(liveNet);
}

function notify(msg, isError, target) {
  const note = target || saveNote;
  note.textContent = msg;
  note.classList.toggle('save-note-error', Boolean(isError));
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

const drawing = initDrawing(canvas, {
  lineWidth: 16,
  onStroke: () => {
    lesson.notify('drawn');
  },
});

async function doInference() {
  if (drawing.isBlank()) { notify('请先写一个数字', true, inferNote); return; }
  try {
    await syncWeights();
    const pixels = drawing.getPixelsGrid();
    const f = await stage.playInference(pixels);
    drawProbBars(f.p);
    const top = f.p.indexOf(Math.max(...f.p));
    notify('模型认为是 ' + top + '（置信度 ' + (f.p[top] * 100).toFixed(1) + '%）', false, inferNote);
    lesson.notify('inference');
  } catch (e) {
    notify('推理失败: ' + (e.message || String(e)), true, inferNote);
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
    notify('已保存到测试集（评估时用于考察模型）', false);
    viewer.refresh(true);
  } catch (e) {
    notify('保存失败: ' + (e.message || String(e)), true);
  }
});

clearBtn.addEventListener('click', () => {
  drawing.clear();
  stage.render();
});

// Boot: init backend then load the real model weights into the stage
panel.init()
  .then(() => syncWeights())
  .catch((e) => notify('初始化失败: ' + (e.message || String(e)), true, inferNote));
