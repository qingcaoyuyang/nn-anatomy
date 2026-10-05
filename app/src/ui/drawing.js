import { preprocess } from '../nn/preprocess.js';

/**
 * Pointer-based handwriting pad. Emits the preprocessed 13x13 grid
 * after every stroke so the UI can show "what the network sees".
 */
export function initDrawing(canvas, { lineWidth = 16, onStroke } = {}) {
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#FFFFFF';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#1F2937';
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  let drawing = false;
  let lastX = 0;
  let lastY = 0;
  let strokes = 0;

  function pos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return [Math.round((e.clientX - rect.left) * scaleX), Math.round((e.clientY - rect.top) * scaleY)];
  }

  function toGray() {
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const gray = new Uint8Array(canvas.width * canvas.height);
    for (let i = 0; i < gray.length; i++) gray[i] = img[i * 4];
    return gray;
  }

  function emit() {
    if (!onStroke) return;
    onStroke(preprocess(toGray(), canvas.width, canvas.height), strokes);
  }

  // Unified handlers: pointer events first, mouse events as fallback.
  // Some WKWebView builds drop pointer events on canvas elements while
  // still delivering classical mouse events, so we listen to both.
  function down(e) {
    e.preventDefault();
    if (e.pointerId != null) {
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch (err) {
        // pointer capture unavailable: fall back to window-level tracking
      }
    }
    drawing = true;
    [lastX, lastY] = pos(e);
    ctx.beginPath();
    ctx.arc(lastX, lastY, lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#1F2937';
    ctx.fill();
    emit();
  }

  function move(e) {
    if (!drawing) return;
    const [x, y] = pos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
  }

  function up() {
    if (!drawing) return;
    drawing = false;
    strokes += 1;
    emit();
  }

  canvas.addEventListener('pointerdown', down);
  canvas.addEventListener('pointermove', move);
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', () => { drawing = false; });

  // Mouse fallback for WKWebView environments without pointer events.
  let sawPointer = false;
  canvas.addEventListener('pointerdown', () => { sawPointer = true; }, { once: true });
  canvas.addEventListener('mousedown', (e) => {
    if (sawPointer) return;
    down(e);
  });
  window.addEventListener('mousemove', (e) => {
    if (sawPointer || !drawing) return;
    move(e);
  });
  window.addEventListener('mouseup', () => {
    if (sawPointer || !drawing) return;
    up();
  });

  return {
    clear() {
      ctx.fillStyle = '#FFFFFF';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      strokes = 0;
      if (onStroke) onStroke(new Array(169).fill(0), 0);
    },
    getPixels13() {
      return preprocess(toGray(), canvas.width, canvas.height);
    },
    isBlank() {
      return strokes === 0;
    },
  };
}
