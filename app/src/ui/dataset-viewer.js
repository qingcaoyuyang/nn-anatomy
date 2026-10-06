import { invoke } from '../tauri-bridge.js';

/**
 * Dataset viewer (bottom of the left column or a stage tab): grid of
 * sample cards showing the actual PNG content, with per-sample delete
 * and one-click import of the bundled 3000/500 dataset.
 */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
}

export function initDatasetViewer(root, opts = {}) {
  const onChange = opts.onChange || (() => {});
  root.innerHTML = '';

  const toolbar = el('div', 'ds-toolbar');
  const splitSel = el('select', 'ds-split');
  for (const [v, name] of [['train', '训练集'], ['test', '测试集']]) {
    const o = el('option', null, name);
    o.value = v;
    splitSel.append(o);
  }
  toolbar.append(splitSel);
  const importBtn = el('button', 'btn btn-primary btn-sm', '导入全量数据集（3000 张）');
  const clearBtn = el('button', 'btn btn-ghost btn-sm', '清空数据集');
  toolbar.append(importBtn, clearBtn);
  const countLabel = el('span', 'ds-count', '0 张');
  toolbar.append(countLabel);
  root.append(toolbar);

  const grid = el('div', 'ds-grid');
  root.append(grid);

  let split = 'train';
  let offset = 0;
  const PAGE = 60;

  async function refresh(reset) {
    if (reset) offset = 0;
    try {
      const cards = await invoke('dataset_list', { split, offset, limit: PAGE });
      if (reset) grid.innerHTML = '';
      for (const c of cards) {
        grid.append(sampleCard(c));
      }
      const stats = await invoke('dataset_stats');
      countLabel.textContent = split === 'train'
        ? `${stats.train_count} 张`
        : `${stats.test_count} 张`;
      onChange(stats);
    } catch (e) {
      grid.append(el('p', 'ds-error', e.message || String(e)));
    }
  }

  function sampleCard(c) {
    const card = el('div', 'ds-card');
    card.dataset.label = c.label;
    const img = el('img', 'ds-img');
    img.src = 'data:image/png;base64,' + c.png_base64;
    img.alt = `样本 ${c.id}（标签 ${c.label}）`;
    card.append(img);
    const tag = el('span', 'ds-tag', String(c.label));
    card.append(tag);
    const del = el('button', 'ds-del', '×');
    del.title = '删除此样本';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      try {
        await invoke('dataset_remove', { split, id: c.id });
        card.remove();
        refresh(false);
      } catch (err) {
        countLabel.textContent = '删除失败: ' + (err.message || String(err));
      }
    });
    card.append(del);
    return card;
  }

  splitSel.addEventListener('change', () => {
    split = splitSel.value;
    refresh(true);
  });

  importBtn.addEventListener('click', async () => {
    importBtn.disabled = true;
    importBtn.textContent = '导入中…';
    try {
      const [n1, n2] = await invoke('dataset_import_builtin');
      countLabel.textContent = '已导入：训练 ' + n1 + ' / 测试 ' + n2;
      refresh(true);
    } catch (e) {
      countLabel.textContent = '导入失败: ' + (e.message || String(e));
    } finally {
      importBtn.disabled = false;
      importBtn.textContent = '导入全量数据集（3000 张）';
    }
  });

  clearBtn.addEventListener('click', async () => {
    clearBtn.disabled = true;
    try {
      await invoke('dataset_clear_all');
      refresh(true);
    } catch (e) {
      countLabel.textContent = '清空失败: ' + (e.message || String(e));
    } finally {
      clearBtn.disabled = false;
    }
  });

  refresh(true);
  return { refresh };
}
