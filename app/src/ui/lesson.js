/**
 * Four-stage lesson stepper driven by real events, never timers.
 *
 * Events: drawn (ink on the pad), inference (forward pass finished),
 * trained (an epoch completed), evaluated (test set scored). Out-of-order
 * events are fine: each marks only its own stage, the highlight always
 * points at the first unfinished stage, and completing all four shows a
 * short cheer. The bar can be dismissed and reset.
 */

const STAGES = [
  { id: 'drawn', title: '像素化', desc: '在画板上写一个数字' },
  { id: 'inference', title: '前向传播', desc: '点「识别」，看信号流过网络' },
  { id: 'trained', title: '训练', desc: '点「训练」，看权重修正错误' },
  { id: 'evaluated', title: '泛化', desc: '评估测试集，看准确率' },
];

export function initLesson(root) {
  if (!root) throw new Error('initLesson 需要引导条容器。');
  root.innerHTML = '';

  root.innerHTML = `
    <ol class="lesson-steps" aria-label="教学引导">
      ${STAGES.map(
        (s, i) => `
        <li class="lesson-step" data-stage="${s.id}">
          <span class="lesson-dot" aria-hidden="true">${i + 1}</span>
          <span class="lesson-text">
            <span class="lesson-title">${s.title}</span>
            <span class="lesson-desc">${s.desc}</span>
          </span>
        </li>`,
      ).join('<li class="lesson-sep" aria-hidden="true">→</li>')}
    </ol>
    <p class="lesson-cheer hidden">🎉 四个阶段都体验过啦，继续自由探索吧！</p>
    <button class="btn btn-ghost btn-sm" id="lesson-skip" type="button">跳过引导</button>`;

  const stepEls = Array.from(root.querySelectorAll('.lesson-step'));
  const cheerEl = root.querySelector('.lesson-cheer');
  const skipBtn = root.querySelector('#lesson-skip');
  const done = new Set();

  function render() {
    const firstOpen = STAGES.findIndex((s) => !done.has(s.id));
    stepEls.forEach((el, i) => {
      const id = STAGES[i].id;
      el.classList.toggle('done', done.has(id));
      el.classList.toggle('active', i === firstOpen);
      const dot = el.querySelector('.lesson-dot');
      dot.textContent = done.has(id) ? '✓' : String(i + 1);
    });
    const allDone = done.size === STAGES.length;
    cheerEl.classList.toggle('hidden', !allDone);
    skipBtn.classList.toggle('hidden', allDone);
  }

  skipBtn.addEventListener('click', () => {
    root.classList.add('hidden');
  });

  render();

  return {
    notify(evt) {
      if (!STAGES.some((s) => s.id === evt)) return;
      if (done.has(evt)) return;
      done.add(evt);
      render();
    },
    reset() {
      done.clear();
      root.classList.remove('hidden');
      render();
    },
  };
}
