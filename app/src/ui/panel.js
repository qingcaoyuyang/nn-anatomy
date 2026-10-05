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
  const onError = hooks.onError || ((msg) => console.error(msg));
  const onModelChange = hooks.onModelChange || (() => {});
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
  const archWarn = el('p', 'arch-warn', '');
  archWarn.style.display = 'none';
  modelBox.append(archWarn);
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
  trainBox.append(el('h3', 'sub-title', '训练（每轮微调一次全部权重）'));
  const epochRow = el('div', 'epoch-row');
  const minusBtn = el('button', 'btn btn-ghost btn-sm', '-');
  const epochVal = el('span', 'epoch-val', '30');
  const plusBtn = el('button', 'btn btn-ghost btn-sm', '+');
  epochRow.append(minusBtn, epochVal, plusBtn);
  trainBox.append(epochRow);
  const oneStep = el('button', 'btn btn-primary', '训练 1 轮');
  const bulkBtn = el('button', 'btn btn-ghost', '连续训练 30 轮');
  trainBox.append(oneStep, bulkBtn);
  root.append(trainBox);

  // --- metrics ---
  const evalBox = el('div', 'sub-box');
  evalBox.append(el('h3', 'sub-title', '评估与指标'));
  const evalBtn = el('button', 'btn btn-ghost', '评估测试集');
  evalBox.append(evalBtn);
  const evalStatus = el('p', 'eval-status', '测试集为空：先保存几张测试样本再评估。');
  evalBox.append(evalStatus);
  const lossCanvas = el('canvas', 'loss-canvas');
  lossCanvas.width = 288;
  lossCanvas.height = 120;
  evalBox.append(lossCanvas);
  const accRow = el('div', 'acc-row');
  const trainAcc = el('span', 'acc-chip', '训练准确率 --');
  const testAcc = el('span', 'acc-chip acc-chip-test', '测试准确率 --');
  accRow.append(trainAcc, testAcc);
  evalBox.append(accRow);
  const perClassBox = el('div', 'per-class-box');
  evalBox.append(perClassBox);
  const confusionBox = el('div', 'confusion-box');
  evalBox.append(confusionBox);
  root.append(evalBox);

  // --- state ---
  let useSgd = false;
  let epochs = 30;
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

  function setEpochs(n) {
    epochs = n;
    epochVal.textContent = epochs;
    bulkBtn.textContent = '连续训练 ' + epochs + ' 轮';
  }
  minusBtn.addEventListener('click', () => setEpochs(Math.max(1, epochs - 1)));
  plusBtn.addEventListener('click', () => setEpochs(Math.min(150, epochs + 1)));

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

  function renderEval(report) {
    if (!report) {
      evalStatus.textContent = '测试集为空：先保存几张测试样本再评估。';
      perClassBox.innerHTML = '';
      confusionBox.innerHTML = '';
      return;
    }
    evalStatus.textContent = `测试准确率 ${(report.acc * 100).toFixed(1)}%（测试集已评估）`;
    perClassBox.innerHTML = '';
    for (let c = 0; c < 10; c++) {
      const row = el('div', 'pc-row');
      const v = report.per_class[c];
      const fill = v == null ? 0 : v;
      row.innerHTML = `<span class="pc-label">${c}</span>
        <span class="pc-bar"><span class="pc-fill" style="width:${(fill * 100).toFixed(1)}%"></span></span>
        <span class="pc-val">${v == null ? '—' : (v * 100).toFixed(0) + '%'}</span>`;
      perClassBox.append(row);
    }
    confusionBox.innerHTML = '';
    let maxCount = 0;
    for (let t = 0; t < 10; t++) {
      for (let p = 0; p < 10; p++) maxCount = Math.max(maxCount, report.confusion[t][p]);
    }
    const table = el('table', 'confusion-table');
    const head = el('tr');
    head.append(el('th'));
    for (let p = 0; p < 10; p++) {
      const th = el('th', null, String(p));
      head.append(th);
    }
    table.append(head);
    for (let t = 0; t < 10; t++) {
      const tr = el('tr');
      const th = el('th', null, String(t));
      tr.append(th);
      for (let p = 0; p < 10; p++) {
        const td = el('td', null, String(report.confusion[t][p]));
        const n = report.confusion[t][p];
        if (n > 0 && maxCount > 0) {
          const a = 0.12 + 0.78 * (n / maxCount);
          td.style.backgroundColor = 'rgba(234,88,12,' + a.toFixed(2) + ')';
          if (n / maxCount > 0.55) td.style.color = '#fff';
        }
        td.title = '真值 ' + t + ' → 预测 ' + p + '：' + n + ' 个';
        tr.append(td);
      }
      table.append(tr);
    }
    const wrap = el('div', 'confusion-wrap');
    const cap = el('p', 'confusion-cap', '行 = 真实数字，列 = 模型预测，对角线 = 答对');
    wrap.append(table, cap);
    confusionBox.append(wrap);
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
    const summary = await invoke('app_init');
    updateArchWarn(summary);
    await refreshModels();
    const h = await invoke('training_history');
    history = h;
    drawLoss();
    if (history.length) updateAcc(history[history.length - 1]);
    try {
      renderEval(await invoke('evaluate_test'));
    } catch (e) {
      renderEval(null);
    }
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
      onError(e.message || String(e));
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
      onError(e.message || String(e));
    } finally {
      bulkBtn.disabled = false;
    bulkBtn.textContent = '连续训练 ' + epochs + ' 轮';
    }
  });

  newModelBtn.addEventListener('click', async () => {
    const count = modelSel.options.length;
    const name = '模型 ' + (count + 1);
    try {
      const summary = await invoke('model_create', { name });
      updateArchWarn(summary);
      history = [];
      drawLoss();
      await refreshModels(name);
      onModelChange(name);
    } catch (e) {
      onError(e.message || String(e));
    }
  });

  evalBtn.addEventListener('click', async () => {
    evalBtn.disabled = true;
    try {
      renderEval(await invoke('evaluate_test'));
      if (hooks.onEvaluate) hooks.onEvaluate();
    } catch (e) {
      renderEval(null);
      evalStatus.textContent = '评估失败：' + (e.message || String(e));
    } finally {
      evalBtn.disabled = false;
    }
  });

  modelSel.addEventListener('change', async () => {
    try {
      const summary = await invoke('model_load', { name: modelSel.value });
      updateArchWarn(summary);
      history = [];
      drawLoss();
      trainAcc.textContent = '训练准确率 --';
      testAcc.textContent = '测试准确率 --';
      onModelChange(modelSel.value);
    } catch (e) {
      onError(e.message || String(e));
    }
  });

  function updateArchWarn(summary) {
    if (summary && summary.arch_mismatch) {
      archWarn.textContent = '此模型为旧版 13×13 输入（169 像素），与当前 16×16 网络不兼容。请新建模型并重新训练。';
      archWarn.style.display = 'block';
    } else {
      archWarn.style.display = 'none';
    }
  }

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
