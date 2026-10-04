import { initDrawing } from './ui/drawing.js';
import { initStage } from './ui/stage.js';
import { createNetwork } from './nn/network.js';

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

function renderSkeleton() {
  const app = document.getElementById('app');
  app.innerHTML = '';
  const shell = el('div', 'app-shell');

  // ===== Header =====
  const header = el('header', 'app-header');
  const titleBox = el('div', 'title-box');
  titleBox.append(el('span', 'logo-dot'));
  titleBox.append(el('h1', null, '神经网络解剖教室'));
  titleBox.append(el('span', 'title-sub', '169 → 16 → 10 · 看见每一次权重更新'));
  header.append(titleBox);
  const stageTabs = el('div', 'stage-tabs', null);
  for (const [key, name] of [['collect', '采集'], ['train', '训练'], ['test', '测试']]) {
    const t = el('button', key === 'collect' ? 'stage-tab active' : 'stage-tab', name);
    t.dataset.stage = key;
    t.addEventListener('click', () => {
      document.querySelectorAll('.stage-tab').forEach((b) => b.classList.toggle('active', b === t));
    });
    stageTabs.append(t);
  }
  header.append(stageTabs);
  shell.append(header);

  // ===== Three columns =====
  const main = el('main', 'columns');

  // --- Left: collect ---
  const left = el('section', 'panel panel-collect');
  left.append(el('h2', 'panel-title', '① 手写采集'));
  left.append(el('p', 'panel-hint', '用鼠标或手指写一个数字（0-9）'));
  const pad = el('div', 'pad-wrap');
  const canvas = el('canvas', 'draw-pad');
  canvas.width = 260;
  canvas.height = 260;
  pad.append(canvas);
  const clearBtn = el('button', 'btn btn-ghost', '清空画板');
  pad.append(clearBtn);
  left.append(pad);

  const gridBox = el('div', 'grid-box');
  const gridTitle = el('h2', 'panel-title', '网络看到的 13×13');
  gridTitle.append(el('span', 'title-sub-2', '重心居中后'));
  gridBox.append(gridTitle);
  const preview = el('canvas', 'grid-preview');
  preview.width = 169;
  preview.height = 169;
  gridBox.append(preview);
  left.append(gridBox);

  const labelRow = el('div', 'label-row', null);
  const labelBtns = [];
  for (let d = 0; d <= 9; d++) {
    const b = el('button', 'digit-btn', String(d));
    b.addEventListener('click', () => {
      labelBtns.forEach((x) => x.classList.toggle('selected', x === b));
    });
    labelBtns.push(b);
    labelRow.append(b);
  }
  left.append(labelRow);

  const saveRow = el('div', 'save-row');
  const toTrain = el('button', 'btn btn-primary', '保存到训练集');
  const toTest = el('button', 'btn btn-ghost', '保存到测试集');
  saveRow.append(toTrain, toTest);
  left.append(saveRow);
  const note = el('p', 'save-note', '训练集需要标注数字；测试集无需标注');
  left.append(note);

  // --- Center: network stage placeholder ---
  const center = el('section', 'panel panel-stage');
  center.append(el('h2', 'panel-title', '② 神经网络舞台'));
  const stageHint = el('p', 'panel-hint', '写完一个数字后松手，信号将逐层传导；点击隐藏节点可查看它的权重热力图');
  center.append(stageHint);
  const stageCanvas = el('canvas', 'stage-canvas');
  stageCanvas.width = 900;
  stageCanvas.height = 560;
  center.append(stageCanvas);

  // --- Right: training placeholder ---
  const right = el('section', 'panel panel-train');
  right.append(el('h2', 'panel-title', '③ 训练面板'));
  right.append(el('div', 'stage-placeholder', 'loss 曲线 / 概率柱 / 混淆矩阵将在此就位'));

  main.append(left, center, right);
  shell.append(main);
  app.append(shell);

  return { canvas, clearBtn, preview, toTrain, toTest, labelBtns, stageCanvas };
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

const ui = renderSkeleton();
const net = createNetwork({ inputs: 169, hidden: 16, outputs: 10 }, 999);
const stage = initStage(ui.stageCanvas, net);
const drawing = initDrawing(ui.canvas, {
  lineWidth: 16,
  onStroke: (pixels13, strokes) => {
    renderPreview(pixels13, ui.preview);
    // play inference when the pen lifts (strokes increments only on pointerup)
    if (strokes > 0) stage.playInference(pixels13);
  },
});
ui.clearBtn.addEventListener('click', () => {
  drawing.clear();
  renderPreview(new Array(169).fill(0), ui.preview);
});
// click a hidden node to inspect its 13x13 weight heatmap
ui.stageCanvas.addEventListener('click', (e) => {
  const rect = ui.stageCanvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (ui.stageCanvas.width / rect.width);
  const y = (e.clientY - rect.top) * (ui.stageCanvas.height / rect.height);
  const H = ui.stageCanvas.height;
  let hit = -1;
  for (let h = 0; h < 16; h++) {
    const ny = H * ((h + 1) / 17);
    if (Math.abs(y - ny) < 14 && Math.abs(x - ui.stageCanvas.width * 0.6) < 20) hit = h;
  }
  stage.highlightNode(1, hit);
});
