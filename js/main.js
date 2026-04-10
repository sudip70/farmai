// main.js — app entry point, UI logic, event handling

import { loadModel, predict }         from './predict.js';
import { fetchDiseaseDetails }        from './diseaseDetails.js';
import { drawOriginal, drawEdges, drawDensity } from './visualize.js';

// ── DOM refs ─────────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const selectedFile  = document.getElementById('selected-file');
const btnAnalyze    = document.getElementById('btn-analyze');
const btnClearUpload = document.getElementById('btn-clear-upload');
const statusDot     = document.getElementById('status-dot');
const statusText    = document.getElementById('status-text');
const spinner       = document.getElementById('spinner');
const uploadSection = document.getElementById('upload-section');
const results       = document.getElementById('results');
const btnReset      = document.getElementById('btn-reset');

const previewImg    = document.getElementById('preview-img');
const predName      = document.getElementById('pred-name');
const confPct       = document.getElementById('conf-pct');
const confFill      = document.getElementById('conf-fill');
const lowConfMsg    = document.getElementById('low-conf-msg');

const detailTitle   = document.getElementById('detail-title');
const detailOv      = document.getElementById('detail-overview');
const detailSy      = document.getElementById('detail-symptoms');
const detailCtrl    = document.getElementById('detail-control');

const vizTabs       = document.getElementById('viz-tabs');
const vizPanel      = document.getElementById('viz-panel');
const canvasOriginal = document.getElementById('canvas-original');
const canvasEdges    = document.getElementById('canvas-edges');
const canvasDensity  = document.getElementById('canvas-density');

let currentObjectUrl = null;
let pendingFile = null;
let modelState = 'loading';
let isAnalyzing = false;

const LOW_CONFIDENCE_THRESHOLD = 30;

// ── Model boot ───────────────────────────────────────────────────────────────
loadModel((state, message) => {
  modelState = state;
  statusDot.className  = `dot ${state}`;
  statusText.textContent = message;
  syncUploadControls();
}).catch(err => {
  console.error('[main] model boot error:', err);
});

// ── Drag & drop ──────────────────────────────────────────────────────────────
dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('over');
  const file = e.dataTransfer.files[0];
  if (file) stageFile(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) stageFile(fileInput.files[0]);
});

btnAnalyze.addEventListener('click', () => {
  if (pendingFile) handleFile(pendingFile);
});

btnClearUpload.addEventListener('click', resetUpload);

// ── Reset ─────────────────────────────────────────────────────────────────────
btnReset.addEventListener('click', () => {
  cleanupPreviewUrl();
  results.style.display   = 'none';
  uploadSection.style.display = '';
  resetUpload();
  confFill.style.width     = '0%';
  setActiveTab('original');
});

// ── Viz tabs ─────────────────────────────────────────────────────────────────
vizTabs.addEventListener('click', e => {
  const tab = e.target.closest('.viz-tab');
  if (!tab) return;
  setActiveTab(tab.dataset.tab);
});

function setActiveTab(name) {
  document.querySelectorAll('.viz-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === name);
  });
  document.querySelectorAll('.viz-pane').forEach(p => {
    p.classList.toggle('active', p.id === `pane-${name}`);
  });
}

function stageFile(file) {
  if (!file.type.startsWith('image/')) {
    alert('Please select a JPG or PNG image.');
    return;
  }

  pendingFile = file;
  selectedFile.textContent = `${file.name} (${formatFileSize(file.size)})`;
  syncUploadControls();
}

function resetUpload() {
  pendingFile = null;
  fileInput.value = '';
  selectedFile.textContent = 'No image selected';
  syncUploadControls();
}

function syncUploadControls() {
  btnAnalyze.disabled = !pendingFile || modelState !== 'ready' || isAnalyzing;
  btnClearUpload.disabled = !pendingFile || isAnalyzing;
}

// ── Main handler ─────────────────────────────────────────────────────────────
async function handleFile(file) {
  if (isAnalyzing) return;

  if (modelState === 'loading') {
    alert('The model is still loading. Please wait a moment and try again.');
    return;
  }
  if (modelState === 'error') {
    alert('The model failed to load. Check the console for details.');
    return;
  }

  cleanupPreviewUrl();
  const objectUrl = URL.createObjectURL(file);
  currentObjectUrl = objectUrl;
  previewImg.src  = objectUrl;
  isAnalyzing = true;
  syncUploadControls();

  uploadSection.style.display = 'none';
  spinner.style.display       = 'block';
  results.style.display       = 'none';

  try {
    const prediction = await predict(objectUrl);
    const vizResults = await Promise.allSettled([
      drawOriginal(canvasOriginal, objectUrl),
      drawEdges(canvasEdges, objectUrl),
      drawDensity(canvasDensity, objectUrl),
    ]);
    const hasVizFailure = vizResults.some(result => result.status === 'rejected');

    renderPrediction(prediction.label, prediction.confidence);
    setActiveTab('original');
    vizPanel.style.display = hasVizFailure ? 'none' : '';
    results.style.display = 'block';

    if (!prediction.isReliable) {
      renderPrediction('Unsupported / uncertain image', prediction.confidence, { lowConfidence: true });
      renderDiseaseDetails(buildUnsupportedDetails(prediction));
      vizPanel.style.display = 'none';
      return;
    }

    const details = await fetchDiseaseDetails(prediction.label);
    renderDiseaseDetails(details);

  } catch (err) {
    console.error('[main] handleFile error:', err);
    renderUiErrorState();
  } finally {
    spinner.style.display = 'none';
    isAnalyzing = false;
    syncUploadControls();
  }
}

// ── Render helpers ────────────────────────────────────────────────────────────
function renderPrediction(label, confidence, { lowConfidence = false } = {}) {
  predName.textContent = label;
  predName.className   = 'prediction-name' + (label === 'Healthy' ? ' healthy' : '');
  confPct.textContent  = `${confidence}%`;

  // Animate bar after a paint frame
  requestAnimationFrame(() => {
    confFill.style.width = `${confidence}%`;
    confFill.className   = `conf-fill${lowConfidence || confidence < LOW_CONFIDENCE_THRESHOLD ? ' low' : ''}`;
  });

  lowConfMsg.textContent = lowConfidence
    ? 'It looks like you uploaded wrong image, please upload image of a leaf.'
    : "Confidence is low — the model couldn't classify this image reliably. Try uploading a clearer, well-lit photo of the leaf.";
  lowConfMsg.style.display = lowConfidence || confidence < LOW_CONFIDENCE_THRESHOLD ? 'block' : 'none';
}

function renderDiseaseDetails({ name, overview, symptoms, control }) {
  detailTitle.textContent = name;
  detailOv.textContent    = overview;
  detailSy.textContent    = symptoms;
  detailCtrl.textContent  = control;
}

function renderUiErrorState() {
  renderPrediction('Wrong image uploaded', 0, { lowConfidence: true });
  renderDiseaseDetails({
    name: 'Please upload a leaf image',
    overview: 'It looks like you uploaded the wrong image. This tool works best with a clear photo of a mango leaf.',
    symptoms: 'Upload one leaf that is clearly visible, fills most of the frame, and has even lighting with minimal background clutter.',
    control: 'Please upload image of a leaf and run the analysis again.',
  });
  confFill.style.width = '0%';
  vizPanel.style.display = 'none';
  setActiveTab('original');
  results.style.display = 'block';
}

function cleanupPreviewUrl() {
  if (!currentObjectUrl) return;
  URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = null;
}

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function buildUnsupportedDetails(prediction) {
  const guidanceByReason = {
    low_confidence: 'The model did not find strong enough evidence for any supported class.',
    ambiguous_result: 'Several classes scored similarly, which usually means the image is too ambiguous for a reliable decision.',
    out_of_domain: 'The image pattern looks outside the model training domain, so the result should not be treated as a disease prediction.',
  };

  return {
    name: 'Unsupported or uncertain image',
    overview: guidanceByReason[prediction.rejectionReason] || 'The uploaded image could not be matched to the supported mango leaf classes with enough certainty.',
    symptoms: 'Upload a single mango leaf with most of the frame occupied by the leaf, even lighting, and minimal background clutter.',
    control: 'Retake the image before using this tool for disease guidance. Do not make treatment decisions from this result alone.',
  };
}
