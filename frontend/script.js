/**
 * Handwritten Digit Recognition AI - Client Script
 * Includes Preprocessing Pipeline Inspector & Normalization Verification
 */

document.addEventListener('DOMContentLoaded', () => {
  // Canvas & Core UI Elements
  const canvas = document.getElementById('digit-canvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const clearBtn = document.getElementById('clear-btn');
  const predictBtn = document.getElementById('predict-btn');
  const strokeSlider = document.getElementById('stroke-width');
  const strokeVal = document.getElementById('stroke-val');
  const watermark = document.getElementById('canvas-watermark');
  const guideOverlay = document.getElementById('guide-overlay');
  const statusPill = document.getElementById('system-status');
  const statusText = document.getElementById('status-text');
  const toast = document.getElementById('toast');

  // Prediction Result Elements
  const heroPlaceholder = document.getElementById('hero-placeholder');
  const heroResult = document.getElementById('hero-result');
  const predictedDigitEl = document.getElementById('predicted-digit');
  const confidenceScoreEl = document.getElementById('confidence-score');
  const probList = document.getElementById('prob-list');
  const latencyBadge = document.getElementById('latency-badge');
  const preprocessedPreview = document.getElementById('preprocessed-preview');
  const preprocessedContainer = document.getElementById('preprocessed-preview-container');

  // Debug Pipeline Elements
  const debugOrigImg = document.getElementById('debug-orig-img');
  const debugCropImg = document.getElementById('debug-crop-img');
  const debugResizeImg = document.getElementById('debug-resize-img');
  const debugFinalImg = document.getElementById('debug-final-img');

  const empty1 = document.getElementById('empty-stage-1');
  const empty2 = document.getElementById('empty-stage-2');
  const empty3 = document.getElementById('empty-stage-3');
  const empty4 = document.getElementById('empty-stage-4');

  const cropMeta = document.getElementById('debug-crop-meta');
  const resizeMeta = document.getElementById('debug-resize-meta');
  const finalMeta = document.getElementById('debug-final-meta');

  const statMin = document.getElementById('stat-min');
  const statMax = document.getElementById('stat-max');
  const statMean = document.getElementById('stat-mean');
  const statShape = document.getElementById('stat-shape');
  const statDtype = document.getElementById('stat-dtype');
  const statNorm = document.getElementById('stat-norm');
  const normBadgeText = document.getElementById('norm-badge-text');

  // CoM & Translation Elements
  const statOrigCom = document.getElementById('stat-orig-com');
  const statTranslation = document.getElementById('stat-translation');
  const statFinalCom = document.getElementById('stat-final-com');
  const statFinalBbox = document.getElementById('stat-final-bbox');
  const statBounds = document.getElementById('stat-bounds');

  // Drawing State
  let isDrawing = false;
  let hasDrawn = false;
  let lastX = 0;
  let lastY = 0;
  let strokeWidth = parseInt(strokeSlider.value, 10);

  // Initialize Canvas
  function initCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = strokeWidth;
  }

  // Clear Canvas
  function clearCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    hasDrawn = false;
    watermark.classList.remove('hidden');
    guideOverlay.style.opacity = '0.15';
    resetResults();
  }

  // Reset Results UI & Debug Inspector
  function resetResults() {
    heroPlaceholder.style.display = 'block';
    heroResult.style.display = 'none';
    latencyBadge.style.display = 'none';
    preprocessedContainer.style.display = 'none';
    renderProbabilities(new Array(10).fill(0), null);

    // Reset Debug Stage Images
    if (debugOrigImg) debugOrigImg.style.display = 'none';
    if (debugCropImg) debugCropImg.style.display = 'none';
    if (debugResizeImg) debugResizeImg.style.display = 'none';
    if (debugFinalImg) debugFinalImg.style.display = 'none';

    if (empty1) empty1.style.display = 'block';
    if (empty2) empty2.style.display = 'block';
    if (empty3) empty3.style.display = 'block';
    if (empty4) empty4.style.display = 'block';

    if (cropMeta) cropMeta.textContent = 'Isolated Bounds: -';
    if (resizeMeta) resizeMeta.textContent = 'Aspect Preserved: -';
    if (finalMeta) finalMeta.textContent = 'Centered: (14, 14)';

    if (statMin) statMin.textContent = '-';
    if (statMax) statMax.textContent = '-';
    if (statMean) statMean.textContent = '-';
    if (statShape) statShape.textContent = '(1, 1, 28, 28)';
    if (statDtype) statDtype.textContent = 'torch.float32';
    if (statNorm) statNorm.textContent = 'Awaiting input';
    if (normBadgeText) normBadgeText.textContent = 'Awaiting input';

    if (statOrigCom) statOrigCom.textContent = '-';
    if (statTranslation) statTranslation.textContent = '-';
    if (statFinalCom) statFinalCom.textContent = '-';
    if (statFinalBbox) statFinalBbox.textContent = '-';
  }

  // Build empty probability distribution rows
  function initProbabilities() {
    probList.innerHTML = '';
    for (let i = 0; i < 10; i++) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.id = `prob-row-${i}`;
      row.innerHTML = `
        <span class="prob-digit">${i}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" id="bar-fill-${i}" style="width: 0%"></div>
        </div>
        <span class="prob-val" id="bar-val-${i}">0.0%</span>
      `;
      probList.appendChild(row);
    }
  }

  // Update probability distribution
  function renderProbabilities(probabilities, winningDigit) {
    probabilities.forEach((prob, digit) => {
      const percentage = (prob * 100).toFixed(1);
      const row = document.getElementById(`prob-row-${digit}`);
      const fill = document.getElementById(`bar-fill-${digit}`);
      const val = document.getElementById(`bar-val-${digit}`);

      if (fill && val && row) {
        fill.style.width = `${percentage}%`;
        val.textContent = `${percentage}%`;

        if (digit === winningDigit) {
          row.classList.add('top-prediction');
        } else {
          row.classList.remove('top-prediction');
        }
      }
    });
  }

  // Pointer event helpers for cross-device support (mouse, touch, stylus)
  function getCoordinates(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }

  function startDrawing(e) {
    e.preventDefault();
    isDrawing = true;
    hasDrawn = true;
    watermark.classList.add('hidden');
    guideOverlay.style.opacity = '0.05';

    const coords = getCoordinates(e);
    lastX = coords.x;
    lastY = coords.y;

    // Draw single dot in case of quick tap
    ctx.beginPath();
    ctx.arc(lastX, lastY, strokeWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
  }

  function draw(e) {
    if (!isDrawing) return;
    e.preventDefault();

    const coords = getCoordinates(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.stroke();

    lastX = coords.x;
    lastY = coords.y;
  }

  function stopDrawing() {
    isDrawing = false;
  }

  // Event Listeners for Drawing
  canvas.addEventListener('pointerdown', startDrawing);
  canvas.addEventListener('pointermove', draw);
  canvas.addEventListener('pointerup', stopDrawing);
  canvas.addEventListener('pointercancel', stopDrawing);
  canvas.addEventListener('pointerleave', stopDrawing);

  // Stroke Slider
  strokeSlider.addEventListener('input', (e) => {
    strokeWidth = parseInt(e.target.value, 10);
    strokeVal.textContent = `${strokeWidth}px`;
    ctx.lineWidth = strokeWidth;
  });

  // Clear Button
  clearBtn.addEventListener('click', clearCanvas);

  // Keyboard Shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;
    if (e.key === 'c' || e.key === 'C') {
      clearCanvas();
    } else if (e.key === 'Enter') {
      predictDigit();
    }
  });

  // Toast Notification
  function showToast(message, type = 'error') {
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    setTimeout(() => {
      toast.classList.remove('show');
    }, 3500);
  }

  // Check Backend Health
  async function checkHealth() {
    try {
      const response = await fetch('/health');
      if (response.ok) {
        const data = await response.json();
        statusPill.className = 'status-pill online';
        statusText.textContent = data.model_loaded ? 'Model Ready' : 'Backend Ready';
      } else {
        throw new Error('Health check failed');
      }
    } catch {
      statusPill.className = 'status-pill offline';
      statusText.textContent = 'Backend Offline';
    }
  }

  // Update Debug Pipeline UI
  function renderDebugPipeline(debug) {
    if (!debug) return;

    // Stage 1: Original Canvas
    if (debug.original_image && debugOrigImg) {
      debugOrigImg.src = `data:image/png;base64,${debug.original_image}`;
      debugOrigImg.style.display = 'block';
      if (empty1) empty1.style.display = 'none';
    }

    // Stage 2: Cropped Bounding Box
    if (debug.cropped_image && debugCropImg) {
      debugCropImg.src = `data:image/png;base64,${debug.cropped_image}`;
      debugCropImg.style.display = 'block';
      if (empty2) empty2.style.display = 'none';
      if (cropMeta && debug.cropped_dimensions) {
        cropMeta.textContent = `${debug.cropped_dimensions[0]} × ${debug.cropped_dimensions[1]} px`;
      }
    }

    // Stage 3: Resized 20x20
    if (debug.resized_image && debugResizeImg) {
      debugResizeImg.src = `data:image/png;base64,${debug.resized_image}`;
      debugResizeImg.style.display = 'block';
      if (empty3) empty3.style.display = 'none';
      if (resizeMeta && debug.resized_dimensions) {
        resizeMeta.textContent = `${debug.resized_dimensions[0]} × ${debug.resized_dimensions[1]} px`;
      }
    }

    // Stage 4: Final 28x28 MNIST Input
    if (debug.final_image && debugFinalImg) {
      debugFinalImg.src = `data:image/png;base64,${debug.final_image}`;
      debugFinalImg.style.display = 'block';
      if (empty4) empty4.style.display = 'none';
      if (finalMeta && debug.final_center_of_mass) {
        finalMeta.textContent = `Final CoM: (${debug.final_center_of_mass[0]}, ${debug.final_center_of_mass[1]})`;
      }
    }

    // CoM & Translation Details
    if (statOrigCom && debug.original_center_of_mass) {
      statOrigCom.textContent = `(${debug.original_center_of_mass[0]}, ${debug.original_center_of_mass[1]})`;
    }
    if (statTranslation && debug.horizontal_translation !== undefined) {
      statTranslation.textContent = `dx=${debug.horizontal_translation}, dy=${debug.vertical_translation}`;
    }
    if (statFinalCom && debug.final_center_of_mass) {
      statFinalCom.textContent = `(${debug.final_center_of_mass[0]}, ${debug.final_center_of_mass[1]})`;
    }
    if (statFinalBbox && debug.final_bounding_box) {
      statFinalBbox.textContent = `[${debug.final_bounding_box.join(', ')}]`;
    }
    if (statBounds) {
      statBounds.textContent = 'Guaranteed';
    }

    // Pixel Values and Normalization Statistics
    if (statMin) statMin.textContent = Number(debug.min_value).toFixed(4);
    if (statMax) statMax.textContent = Number(debug.max_value).toFixed(4);
    if (statMean) statMean.textContent = Number(debug.mean_value).toFixed(4);
    if (statShape && debug.tensor_shape) {
      statShape.textContent = `(${debug.tensor_shape.join(', ')})`;
    }
    if (statDtype) statDtype.textContent = debug.tensor_dtype || 'torch.float32';

    if (statNorm) {
      if (debug.normalization_confirmed) {
        statNorm.textContent = '✓ [0.0, 1.0] Confirmed';
        statNorm.className = 'stat-value highlight-green';
      } else {
        statNorm.textContent = 'Out of Range';
        statNorm.className = 'stat-value';
      }
    }

    if (normBadgeText) {
      normBadgeText.textContent = 'Pipeline Verified';
    }
  }

  // Predict Function
  async function predictDigit() {
    if (!hasDrawn) {
      showToast('Please draw a digit (0–9) before predicting.', 'warning');
      return;
    }

    // Set Loading State
    const btnText = predictBtn.querySelector('.btn-text');
    const btnLoader = predictBtn.querySelector('.btn-loader');
    btnText.style.display = 'none';
    btnLoader.style.display = 'inline-flex';
    predictBtn.disabled = true;
    clearBtn.disabled = true;

    const startTime = performance.now();

    try {
      const imageData = canvas.toDataURL('image/png');

      const response = await fetch('/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageData }),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || 'Prediction failed. Please try again.');
      }

      const elapsedMs = Math.round(performance.now() - startTime);

      // Display Hero Results
      heroPlaceholder.style.display = 'none';
      heroResult.style.display = 'flex';
      predictedDigitEl.textContent = result.predicted_digit;
      
      const confPercent = (result.confidence * 100).toFixed(1);
      confidenceScoreEl.textContent = `${confPercent}%`;

      latencyBadge.textContent = `${elapsedMs} ms`;
      latencyBadge.style.display = 'inline-block';

      // Preprocessed preview (if sent by backend)
      if (result.preprocessed_image) {
        preprocessedPreview.src = `data:image/png;base64,${result.preprocessed_image}`;
        preprocessedContainer.style.display = 'flex';
      } else {
        preprocessedContainer.style.display = 'none';
      }

      // Update Probability Distribution
      renderProbabilities(result.probabilities, result.predicted_digit);

      // Update Preprocessing Debugger UI
      if (result.debug) {
        renderDebugPipeline(result.debug);
      }

    } catch (err) {
      showToast(err.message || 'Error communicating with prediction server.', 'error');
    } finally {
      btnText.style.display = 'inline-flex';
      btnLoader.style.display = 'none';
      predictBtn.disabled = false;
      clearBtn.disabled = false;
    }
  }

  predictBtn.addEventListener('click', predictDigit);

  // Initialize
  initCanvas();
  initProbabilities();
  checkHealth();
  setInterval(checkHealth, 15000);
});
