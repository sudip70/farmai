// visualize.js — canvas visualizations: Sobel edge detection + RGB histogram

const VIZ_WIDTH = 760; // internal canvas resolution
const VIZ_MAX_HEIGHT = 500;
const DENSITY_HEIGHT = 300;

/**
 * Draw the original image scaled to fit the canvas.
 */
export function drawOriginal(canvas, imgSrc) {
  return withImage(imgSrc, img => {
    const { w, h } = fitSize(img.naturalWidth, img.naturalHeight);
    canvas.width  = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);
  });
}

/**
 * Sobel edge detection.
 * Converts to grayscale, applies Gx + Gy kernels, renders magnitude
 * as a green-tinted heatmap on a dark background.
 */
export function drawEdges(canvas, imgSrc) {
  return withImage(imgSrc, img => {
    const { w, h } = fitSize(img.naturalWidth, img.naturalHeight);
    canvas.width  = w;
    canvas.height = h;

    // 1. Draw image to off-screen canvas to extract pixel data
    const offscreen = document.createElement('canvas');
    offscreen.width  = w;
    offscreen.height = h;
    const offCtx = offscreen.getContext('2d');
    offCtx.drawImage(img, 0, 0, w, h);

    const srcData  = offCtx.getImageData(0, 0, w, h).data;
    const gray     = toGrayscale(srcData, w, h);

    // 2. Sobel
    const mag = sobelMagnitude(gray, w, h);

    // 3. Normalise to [0,255]
    const max = arrayMax(mag);
    const norm = new Float32Array(mag.length);
    if (max > 0) {
      for (let i = 0; i < mag.length; i++) {
        norm[i] = mag[i] / max * 255;
      }
    }

    // 4. Paint onto output canvas
    const outCtx = canvas.getContext('2d');
    const outImg  = outCtx.createImageData(w, h);

    for (let i = 0; i < norm.length; i++) {
      const v = norm[i];
      // Green-tinted: dim background, bright accent on edges
      outImg.data[i * 4 + 0] = v * 0.4;          // R
      outImg.data[i * 4 + 1] = v * 0.85;         // G — dominant
      outImg.data[i * 4 + 2] = v * 0.5;          // B
      outImg.data[i * 4 + 3] = 255;              // A
    }

    outCtx.putImageData(outImg, 0, 0);
  });
}

/**
 * RGB channel histogram (256 bins each).
 * Drawn as overlapping semi-transparent filled curves on a dark canvas.
 */
export function drawDensity(canvas, imgSrc) {
  return withImage(imgSrc, img => {
    const W = VIZ_WIDTH;
    const H = DENSITY_HEIGHT;
    canvas.width  = W;
    canvas.height = H;

    // 1. Sample pixels from a small off-screen version (fast)
    const sample = 320;
    const off    = document.createElement('canvas');
    off.width = off.height = sample;
    const offCtx = off.getContext('2d');
    offCtx.drawImage(img, 0, 0, sample, sample);
    const data = offCtx.getImageData(0, 0, sample, sample).data;

    // 2. Build histograms
    const bins = 256;
    const rHist = new Float32Array(bins);
    const gHist = new Float32Array(bins);
    const bHist = new Float32Array(bins);

    for (let i = 0; i < data.length; i += 4) {
      rHist[data[i]]     += 1;
      gHist[data[i + 1]] += 1;
      bHist[data[i + 2]] += 1;
    }

    const maxVal = Math.max(arrayMax(rHist), arrayMax(gHist), arrayMax(bHist));

    // 3. Draw
    const ctx     = canvas.getContext('2d');
    const pad     = { top: 20, right: 20, bottom: 40, left: 50 };
    const plotW   = W - pad.left - pad.right;
    const plotH   = H - pad.top  - pad.bottom;

    // Background
    ctx.fillStyle = '#0d0f0e';
    ctx.fillRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth   = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + plotH * (1 - i / 4);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
    }

    // Draw each channel as a filled area curve
    const channels = [
      { hist: rHist, fill: 'rgba(220,90,70,0.35)',  stroke: 'rgba(220,90,70,0.9)' },
      { hist: gHist, fill: 'rgba(110,200,110,0.35)', stroke: 'rgba(110,200,110,0.9)' },
      { hist: bHist, fill: 'rgba(80,140,220,0.35)',  stroke: 'rgba(80,140,220,0.9)' },
    ];

    channels.forEach(({ hist, fill, stroke }) => {
      ctx.beginPath();
      // Start from bottom-left
      ctx.moveTo(pad.left, pad.top + plotH);

      for (let b = 0; b < bins; b++) {
        const x = pad.left + (b / (bins - 1)) * plotW;
        const y = pad.top  + plotH * (1 - hist[b] / maxVal);
        b === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }

      // Close to bottom
      ctx.lineTo(pad.left + plotW, pad.top + plotH);
      ctx.lineTo(pad.left,         pad.top + plotH);
      ctx.closePath();

      ctx.fillStyle = fill;
      ctx.fill();

      // Redraw just the top stroke
      ctx.beginPath();
      for (let b = 0; b < bins; b++) {
        const x = pad.left + (b / (bins - 1)) * plotW;
        const y = pad.top  + plotH * (1 - hist[b] / maxVal);
        b === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.strokeStyle = stroke;
      ctx.lineWidth   = 1.5;
      ctx.stroke();
    });

    // Axes
    ctx.strokeStyle = 'rgba(255,255,255,0.2)';
    ctx.lineWidth   = 1;
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    ctx.lineTo(pad.left, pad.top + plotH);
    ctx.lineTo(pad.left + plotW, pad.top + plotH);
    ctx.stroke();

    // X-axis labels: 0, 64, 128, 192, 255
    ctx.fillStyle = 'rgba(255,255,255,0.35)';
    ctx.font      = `${Math.round(W * 0.014)}px Epilogue, sans-serif`;
    ctx.textAlign = 'center';
    [0, 64, 128, 192, 255].forEach(v => {
      const x = pad.left + (v / 255) * plotW;
      ctx.fillText(v, x, pad.top + plotH + 20);
    });

    // Y label
    ctx.save();
    ctx.translate(14, H / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.fillText('Frequency', 0, 0);
    ctx.restore();

    // Legend
    const legendItems = [
      { label: 'Red',   color: 'rgba(220,90,70,0.9)' },
      { label: 'Green', color: 'rgba(110,200,110,0.9)' },
      { label: 'Blue',  color: 'rgba(80,140,220,0.9)' },
    ];
    const lx = pad.left + plotW - 110;
    const ly = pad.top + 12;
    legendItems.forEach(({ label, color }, i) => {
      ctx.fillStyle = color;
      ctx.fillRect(lx, ly + i * 18, 10, 10);
      ctx.fillStyle = 'rgba(255,255,255,0.5)';
      ctx.textAlign = 'left';
      ctx.font      = `${Math.round(W * 0.013)}px Epilogue, sans-serif`;
      ctx.fillText(label, lx + 14, ly + i * 18 + 9);
    });
  });
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Scale image to fit within VIZ_WIDTH maintaining aspect ratio.
 */
function fitSize(natW, natH) {
  const scale = Math.min(1, VIZ_WIDTH / natW, VIZ_MAX_HEIGHT / natH);
  return { w: Math.round(natW * scale), h: Math.round(natH * scale) };
}

/**
 * Returns a Promise that resolves once the image is loaded,
 * then calls fn(img) and resolves with its return value.
 */
function withImage(src, fn) {
  return new Promise((res, rej) => {
    const img  = new Image();
    img.onload = () => { try { res(fn(img)); } catch (e) { rej(e); } };
    img.onerror = () => rej(new Error('The selected file could not be decoded as an image.'));
    img.src = src;
  });
}

function arrayMax(values) {
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > max) max = values[i];
  }
  return max;
}

/**
 * Convert RGBA pixel array to a flat Float32Array of grayscale values.
 */
function toGrayscale(data, w, h) {
  const gray = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const base = i * 4;
    gray[i] = 0.299 * data[base] + 0.587 * data[base + 1] + 0.114 * data[base + 2];
  }
  return gray;
}

/**
 * Apply Sobel operator and return magnitude array.
 * Edge pixels return 0 (no padding needed for the full image).
 */
function sobelMagnitude(gray, w, h) {
  const mag = new Float32Array(w * h);

  // Gx and Gy kernels (standard 3×3 Sobel)
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const idx = y * w + x;

      const tl = gray[(y-1)*w+(x-1)];  const t  = gray[(y-1)*w+x];  const tr = gray[(y-1)*w+(x+1)];
      const ml = gray[ y   *w+(x-1)];                                 const mr = gray[ y   *w+(x+1)];
      const bl = gray[(y+1)*w+(x-1)];  const b  = gray[(y+1)*w+x];  const br = gray[(y+1)*w+(x+1)];

      const gx = -tl - 2*ml - bl + tr + 2*mr + br;
      const gy = -tl - 2*t  - tr + bl + 2*b  + br;

      mag[idx] = Math.sqrt(gx*gx + gy*gy);
    }
  }
  return mag;
}
