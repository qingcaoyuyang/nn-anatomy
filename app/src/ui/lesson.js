/**
 * Four-stage lesson guide: pixelation -> forward -> training ->
 * generalization. Each stage lights up when its milestone is reached.
 */

const STAGES = [
  { key: 'collect', name: '① 采集', hint: '手写几个数字，存入训练集', done: false },
  { key: 'forward', name: '② 前向', hint: '松手后看信号逐层传导', done: false },
  { key: 'train', name: '③ 训练', hint: '看权重一步步调整', done: false },
  { key: 'generalize', name: '④ 泛化', hint: '用没见过的数字考考它', done: false },
];

export function initLesson(root) {
  root.innerHTML = '';
  const bar = document.createElement('div');
  bar.className = 'lesson-bar';
  const nodes = [];
  for (const s of STAGES) {
    const item = document.createElement('div');
    item.className = 'lesson-item';
    const name = document.createElement('span');
    name.className = 'lesson-name';
    name.textContent = s.name;
    const hint = document.createElement('span');
    hint.className = 'lesson-hint';
    hint.textContent = s.hint;
    item.append(name, hint);
    bar.append(item);
    nodes.push(item);
  }
  root.append(bar);

  function mark(stageKey) {
    const idx = STAGES.findIndex((s) => s.key === stageKey);
    if (idx < 0) return;
    STAGES[idx].done = true;
    nodes.forEach((n, i) => {
      n.classList.toggle('done', i <= idx);
    });
  }

  function current() {
    return STAGES.filter((s) => s.done).length;
  }

  return { mark, current };
}
