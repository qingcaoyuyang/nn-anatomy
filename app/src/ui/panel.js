import { invoke } from '../tauri-bridge.js';

/**
 * Training control panel (right column): model management, optimizer
 * settings, epoch controls, loss curve and accuracy readouts.
 */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

export function initPanel(root, hooks = {}) {
  const onEpoch = hooks.onEpoch || (() => {});
  root.innerHTML = '';

  // --- model row ---
  const modelBox = el('div', 'sub-box');
  modelBox.append(el('h3', 'sub-title', '模型'));
  const modelRow = el('div', 'model-row');
  const modelSel = el('select', 'model-select');
  modelRow.append(modelSel);
  const newModelBtn = el('button', 'btn btn-ghost btn-sm', '新建');
  modelRow.append(newModelBtn);
  modelBox.append(modelRow);
  root.append(modelBox);

  // --- optimizer row ---
  const optBox = el('div', 'sub-box');
  optBox.append(el('h3', 'sub-title', '优化器'));
  const optRow = el('div', 'opt-row');
  const adamBtn = el('button', 'opt-btn active', 'Adam');
  const sgdBtn = el('button', 'opt-btn', 'SGD');
  optRow.append(adamBtn, sgdBtn);
  optBox.append(optRow);
  const lrRow = el('div', 'lr-row');
  lrRow.append(el('span', 'lr-label', '学习率'));
  const lrVal = el('span', 'lr-val', '0.001');
  lrRow.append(lrVal);
  optBox.append(lrRow);
  const lrSlider = el('input', 'lr-slider');
  lrSlider.type = 'range';
  lrSlider.min = '1';
  lrSlider.max = '100';
  lrSlider.value = '1';
  optBox.append(lrSlider);
  root.append(optBox);

  // --- training controls ---
  const trainBox = el('div', 'sub-box');
  trainBox.append(el('h3', 'sub-title', '训练'));
  const epochRow = el('div', 'epoch-row');
  const minusBtn = el('button', 'btn btn-ghost btn-sm', '-');
  const epochVal = el('span', 'epoch-val', '1');
  const plusBtn = el('button', 'btn btn-ghost btn-sm', '+');
  epochRow.append(minusBtn, epochVal, plusBtn);
  trainBox.append(epochRow);
  const oneStep = el('button', 'btn btn-primary', '单步训练');
  const bulkBtn = el('button', 'btn btn-ghost', '批量快进');
  trainBox.append(oneStep, bulkBtn);
  root.append(trainBox);

  // --- metrics ---
  const metricBox = el('div', 'sub-box');
  metricBox.append(el('h3', 'sub-title', '指标'));
  const lossCanvas = el('canvas', 'loss-canvas');
  lossCanvas.width = 288;
  lossCanvas.height = 120;
  metricBox.append(lossCanvas);
  const accRow = el('div', 'acc-row');
  const trainAcc = el('span', 'acc-chip', '训练准确率 --');
  const testAcc = el('span', 'acc-chip acc-chip-test', '测试准确率 --');
  accRow.append(trainAcc, testAcc);
  metricBox.append(accRow);
  root.append(metricBox);

  // --- state ---
  let useSgd = false;
  let epochs = 1;
  let lr = 0.001;
  let history = [];

  function currentLr() {
    // slider 1..100 maps logarithmically to 0.0001..0.1
    return Math.round(Math.pow(10, -4 + (lrSlider.value / 100) * 3) * 10000) / 10000;
  }
  lrSlider.addEventListener('input', () => {
    lr = currentLr();
    lrVal.textContent = lr.toFixed(4);
  });

  adamBtn.addEventListener('click', () => {
    useSgd = false;
    adamBtn.classList.add('active');
    sgdBtn.classList.remove('active');
  });
  sgdBtn.addEventListener('click', () => {
    useSgd = true;
    sgdBtn.classList.add('active');
    adamBtn.classList.remove('active');
  });

  minusBtn.addEventListener('click', () => { epochs = Math.max(1, epochs - 1); epochVal.textContent = epochs; });
  plusBtn.addEventListener('click', () => { epochs = Math.min(150, epochs + 1); epochVal.textContent = epochs; });

  function drawLoss() {
    const ctx = lossCanvas.getContext('2d');
    ctx.clearRect(0, 0, lossCanvas.width, lossCanvas.height);
    if (history.length < 2) {
      ctx.fillStyle = '#92400E';
      ctx.font = '12px sans-serif';
      ctx.fillText('训练后 loss 曲线将出现在这里', 12, 60);
      return;
    }
    const maxLoss = Math.max(...history.map((h) => h.loss), 0.001);
    const W = lossCanvas.width - 20;
    const H = lossCanvas.height - 20;
    ctx.strokeStyle = '#F97316';
    ctx.lineWidth = 2;
    ctx.beginPath();
    history.forEach((h, i) => {
      const x = 10 + (i / (history.length - 1)) * W;
      const y = 10 + (1 - h.loss / maxLoss) * H;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    // baseline grid
    ctx.strokeStyle = 'rgba(254,215,170,0.5)';
    ctx.lineWidth = 0.5;
    for (let g = 1; g < 4; g++) {
      ctx.beginPath();
      ctx.moveTo(10, 10 + (g / 4) * H);
      ctx.lineTo(10 + W, 10 + (g / 4) * H);
      ctx.stroke();
    }
  }

  function updateAcc(point) {
    trainAcc.textContent = `训练准确率 ${(point.acc * 100).toFixed(1)}%`;
    testAcc.textContent = `测试准确率 ${(point.test_acc * 100).toFixed(1)}%`;
  }

  async function refreshModels(current) {
    const names = await invoke('model_list');
    modelSel.innerHTML = '';
    for (const n of names) {
      const opt = el('option', null, n);
      opt.value = n;
      modelSel.append(opt);
    }
    if (current) modelSel.value = current;
  }

  // Boot: initialize backend workspace state (app_init) before first use.
  async function boot() {
    await invoke('app_init');
    await refreshModels();
  }

  oneStep.addEventListener('click', async () => {
    oneStep.disabled = true;
    try {
      const point = await invoke('train_one_epoch', { lr, useSgd });
      history.push(point);
      drawLoss();
      updateAcc(point);
      onEpoch(point);
    } catch (e) {
      alert(e.message || String(e));
    } finally {
      oneStep.disabled = false;
    }
  });

  bulkBtn.addEventListener('click', async () => {
    bulkBtn.disabled = true;
    bulkBtn.textContent = '训练中…';
    try {
      const points = await invoke('train_bulk', { epochs, lr, useSgd });
      history = history.concat(points);
      drawLoss();
      if (points.length) updateAcc(points[points.length - 1]);
      onEpoch(points[points.length - 1]);
    } catch (e) {
      alert(e.message || String(e));
    } finally {
      bulkBtn.disabled = false;
      bulkBtn.textContent = '批量快进';
    }
  });

  newModelBtn.addEventListener('click', async () => {
    const name = prompt('新模型名称：', '');
    if (!name) return;
    try {
      await invoke('model_create', { name });
      history = [];
      drawLoss();
      await refreshModels(name);
    } catch (e) {
      alert(e.message || String(e));
    }
  });

  modelSel.addEventListener('change', async () => {
    try {
      await invoke('model_load', { name: modelSel.value });
      history = [];
      drawLoss();
      trainAcc.textContent = '训练准确率 --';
      testAcc.textContent = '测试准确率 --';
    } catch (e) {
      alert(e.message || String(e));
    }
  });

  drawLoss();
  return {
    async init() {
      await boot();
    },
    pushEpoch(point) {
      history.push(point);
      drawLoss();
      updateAcc(point);
    },
    getHistory() { return history; },
  };
}
