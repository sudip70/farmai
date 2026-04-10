// predict.js — model loading + inference

const MODEL_URL = new URL('../model/model.json', import.meta.url).href;
const METADATA_URL = new URL('../model/metadata.json', import.meta.url).href;
const EFFICIENTNET_MEAN = tf.tensor1d([0.485, 0.456, 0.406]);
const EFFICIENTNET_STD = tf.tensor1d([
  Math.sqrt(0.229),
  Math.sqrt(0.224),
  Math.sqrt(0.225),
]);
const FALLBACK_CLASS_LABELS = [
  'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
  'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
];
const MIN_SUPPORTED_CONFIDENCE = 55;
const MIN_SUPPORTED_MARGIN = 12;
const MAX_SUPPORTED_ENTROPY = 0.82;

let model = null;
let classLabels = [...FALLBACK_CLASS_LABELS];
let preprocessingMode = 'efficientnet_rgb_mean_std';

/**
 * Download and warm up the TF.js model.
 * Calls onStatus(state, message) where state is 'loading' | 'ready' | 'error'.
 */
export async function loadModel(onStatus) {
  onStatus('loading', 'Loading model…');
  try {
    await loadMetadata();
    model = await tf.loadLayersModel(MODEL_URL);
    // Warm-up pass so the first real prediction isn't slow
    const dummy = tf.zeros([1, 224, 224, 3]);
    const warmup = model.predict(dummy);
    if (Array.isArray(warmup)) {
      warmup.forEach(tensor => tensor.dispose());
    } else {
      warmup.dispose();
    }
    dummy.dispose();
    onStatus('ready', `Model ready (${classLabels.length} classes)`);
  } catch (err) {
    const reason = err?.message || String(err);
    onStatus('error', `Model failed to load — ${reason}`);
    console.error('[predict] loadModel error:', err);
    throw err;
  }
}

/**
 * Run EfficientNetB0 inference on an img element or object URL string.
 * Returns { label, confidence, allScores }.
 */
export async function predict(imgSource) {
  if (!model) throw new Error('Model is not loaded yet.');

  const img = await resolveImage(imgSource);

  const rawScores = tf.tidy(() => {
    const tensor = tf.browser.fromPixels(img)
      .resizeBilinear([224, 224])
      .toFloat();

    const prepared = preprocessingMode === 'model_embedded'
      ? tensor.expandDims(0)
      : tensor
          // Match the original EfficientNet preprocessing layers removed at export.
          .div(255)
          .sub(EFFICIENTNET_MEAN)
          .div(EFFICIENTNET_STD)
          .expandDims(0);

    return model.predict(prepared);
  });
  const scores    = await rawScores.data();
  rawScores.dispose();

  const scoreList = Array.from(scores);
  const rankedScores = [...scoreList].sort((a, b) => b - a);
  const bestScore = rankedScores[0] ?? 0;
  const secondBestScore = rankedScores[1] ?? 0;
  const maxIdx = scoreList.indexOf(bestScore);
  const label = classLabels[maxIdx] || `Class ${maxIdx}`;
  const confidence = Math.round(bestScore * 100);
  const margin = Math.round((bestScore - secondBestScore) * 100);
  const entropy = normalizedEntropy(scoreList);
  const isReliable =
    confidence >= MIN_SUPPORTED_CONFIDENCE &&
    margin >= MIN_SUPPORTED_MARGIN &&
    entropy <= MAX_SUPPORTED_ENTROPY;

  let rejectionReason = null;
  if (!isReliable) {
    if (confidence < MIN_SUPPORTED_CONFIDENCE) {
      rejectionReason = 'low_confidence';
    } else if (margin < MIN_SUPPORTED_MARGIN) {
      rejectionReason = 'ambiguous_result';
    } else {
      rejectionReason = 'out_of_domain';
    }
  }

  return {
    label,
    confidence,
    margin,
    entropy,
    isReliable,
    rejectionReason,
    allScores: scoreList,
  };
}

// Helpers ────────────────────────────────────────────────────────────────────

async function loadMetadata() {
  try {
    const response = await fetch(METADATA_URL, { cache: 'no-store' });
    if (!response.ok) return;

    const metadata = await response.json();
    if (Array.isArray(metadata.class_labels) && metadata.class_labels.length > 0) {
      classLabels = metadata.class_labels;
    }
    if (typeof metadata.preprocessing === 'string') {
      preprocessingMode = metadata.preprocessing;
    }
  } catch (err) {
    console.warn('[predict] metadata load skipped:', err);
  }
}

function resolveImage(src) {
  if (src instanceof HTMLImageElement) {
    return src.complete
      ? Promise.resolve(src)
      : new Promise((res, rej) => { src.onload = () => res(src); src.onerror = rej; });
  }
  return new Promise((res, rej) => {
    const img  = new Image();
    img.onload = () => res(img);
    img.onerror = () => rej(new Error('The selected file could not be decoded as an image.'));
    img.src = src;
  });
}

function normalizedEntropy(scores) {
  const epsilon = 1e-12;
  const entropy = scores.reduce((sum, score) => {
    const safeScore = Math.max(score, epsilon);
    return sum - safeScore * Math.log(safeScore);
  }, 0);

  return entropy / Math.log(scores.length || 1);
}
