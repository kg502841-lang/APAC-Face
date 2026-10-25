/**
 * Faceify AI Facial Landmark Analysis - Master Controller
 * --------------------------------------------------------
 * Orchestrates webcam feeds, image uploads, the MediaPipe baseline and the trained APAC-Face model,
 * ground-truth error metrics for both models, the benchmark gallery, the measured benchmark
 * results, the coordinates table and high-DPI canvas visualization.
 */

(function () {
  'use strict';

  const state = {
    activeSource: 'upload',     // 'webcam' | 'upload'
    activeModelKey: 'apac',     // 'baseline' | 'apac'
    compareMode: false,

    zoomLevel: 1.0,
    panX: 0,
    panY: 0,

    showLandmarks: true,
    showConnections: true,
    showPredictions: true,
    showGroundTruth: true,
    showErrorVectors: true,

    currentImageElement: null,
    currentVideoElement: null,
    currentGroundTruth: null,
    currentSampleKey: null,
    results: { baseline: null, apac: null },   // latest detection result of each model
    metrics: { baseline: null, apac: null },   // ground-truth metrics of each model (null without GT)

    webcamStream: null,
    animFrameId: null,
    isProcessingFrame: false,
    frameCount: 0,

    baselineDetector: null,
    apacDetector: null,
    datasetManager: null,
    benchmark: null,

    galleryFilter: 'all',
    galleryQuery: '',
    galleryPage: 1,
    galleryPageSize: 12
  };

  const MODEL_LABEL = { baseline: "Google MediaPipe", apac: "APAC-Face (trained)" };
  const UI = {};

  function initDOMElements() {
    UI.app = document.getElementById('faceify-landmark-app');
    UI.tabWebcam = document.getElementById('fcf-tab-webcam');
    UI.tabUpload = document.getElementById('fcf-tab-upload');
    UI.tabCapture = document.getElementById('fcf-tab-capture');
    UI.captureBar = document.getElementById('fcf-capture-bar');
    UI.btnCapture = document.getElementById('fcf-btn-capture');
    UI.btnCaptureCancel = document.getElementById('fcf-btn-capture-cancel');
    UI.viewportArea = document.querySelector('.fcf-viewport-area');
    UI.dropzone = document.getElementById('fcf-dropzone');
    UI.fileInput = document.getElementById('fcf-file-input');

    UI.stageSingle = document.getElementById('fcf-stage-single');
    UI.stageCompare = document.getElementById('fcf-stage-compare');
    UI.mainCanvas = document.getElementById('fcf-main-canvas');
    UI.videoEl = document.getElementById('fcf-video-stream');
    UI.imgEl = document.getElementById('fcf-image-holder');
    UI.canvasCompareBaseline = document.getElementById('fcf-compare-canvas-baseline');
    UI.canvasCompareApac = document.getElementById('fcf-compare-canvas-apac');

    UI.modelSelectBaseline = document.getElementById('fcf-model-btn-baseline');
    UI.modelSelectApac = document.getElementById('fcf-model-btn-apac');
    UI.compareToggle = document.getElementById('fcf-compare-checkbox');
    UI.modelStatus = document.getElementById('fcf-model-status');

    UI.metricLandmarksCount = document.getElementById('fcf-val-landmarks-count');
    UI.metricFaceStatus = document.getElementById('fcf-val-face-status');
    UI.metricProcessingTime = document.getElementById('fcf-val-processing-time');
    UI.metricStability = document.getElementById('fcf-val-stability');
    UI.metricModelName = document.getElementById('fcf-val-model-name');

    UI.gtMeanError = document.getElementById('fcf-gt-mean-error');
    UI.gtNme = document.getElementById('fcf-gt-nme');
    UI.gtMaxError = document.getElementById('fcf-gt-max-error');
    UI.gtFailureRate = document.getElementById('fcf-gt-failure-rate');
    UI.gtPointsCompared = document.getElementById('fcf-gt-points-compared');
    UI.gtStatusNotice = document.getElementById('fcf-gt-status-notice');
    UI.gtFormulaDetails = document.getElementById('fcf-gt-formula-details');
    UI.gtModelLabel = document.getElementById('fcf-gt-model-label');
    UI.gtCompareBody = document.getElementById('fcf-gt-compare-tbody');
    UI.gtCompareSummary = document.getElementById('fcf-gt-compare-summary');

    UI.chkShowLandmarks = document.getElementById('fcf-chk-landmarks');
    UI.chkShowMesh = document.getElementById('fcf-chk-mesh');
    UI.chkShowPredictions = document.getElementById('fcf-chk-pred');
    UI.chkShowGroundTruth = document.getElementById('fcf-chk-gt');
    UI.chkShowErrorVectors = document.getElementById('fcf-chk-vectors');

    UI.btnZoomIn = document.getElementById('fcf-btn-zoom-in');
    UI.btnZoomOut = document.getElementById('fcf-btn-zoom-out');
    UI.btnResetZoom = document.getElementById('fcf-btn-reset-zoom');

    UI.coordsTableBody = document.getElementById('fcf-coords-tbody');
    UI.coordsSearchInput = document.getElementById('fcf-coords-search');
    UI.btnExportCsv = document.getElementById('fcf-btn-export-csv');
    UI.coordsCountLabel = document.getElementById('fcf-coords-count');

    UI.galleryContainer = document.getElementById('fcf-gallery-container');
    UI.annotationFileInput = document.getElementById('fcf-annotation-file-input');

    UI.mobileMenuToggle = document.getElementById('fcf-mobile-toggle');
    UI.navLinks = document.getElementById('fcf-nav-links');
  }

  async function initApp() {
    initDOMElements();
    setupEventListeners();
    setupHeroCanvas();
    setModelStatus("Loading Google MediaPipe and the trained APAC-Face model ...", "amber");

    state.datasetManager = new DatasetManager();
    state.baselineDetector = new MediaPipeDetector();
    state.apacDetector = new APACDetector();

    const [, mpOk, apacOk] = await Promise.all([
      state.datasetManager.loadFullCatalog().then(() => { updateGalleryTabCounts(); renderBenchmarkGallery(); }),
      state.baselineDetector.initialize(),
      state.apacDetector.initialize(),
      loadBenchmarkResults()
    ]);

    const problems = [];
    if (!mpOk) problems.push(state.baselineDetector.loadError);
    if (!apacOk) problems.push(state.apacDetector.loadError);
    if (problems.length) {
      setModelStatus(problems.join(" | "), "crimson");
    } else {
      setModelStatus(`Both models running on this device: ${state.baselineDetector.getModelInfo().statusMessage} · ` +
                     `${state.apacDetector.getModelInfo().statusMessage}.`, "emerald");
    }

    const first = state.datasetManager.wflwSamples[0];
    if (first) loadBenchmarkSample(first.key, first.imagePath, false);
  }

  function setModelStatus(text, tone) {
    if (!UI.modelStatus) return;
    UI.modelStatus.textContent = text;
    UI.modelStatus.className = `fcf-model-status fcf-model-status-${tone}`;
  }

  function setupEventListeners() {
    if (UI.mobileMenuToggle && UI.navLinks) {
      UI.mobileMenuToggle.addEventListener('click', () => UI.navLinks.classList.toggle('mobile-open'));
    }

    UI.tabWebcam.addEventListener('click', () => switchSource('webcam'));
    // "Upload Photo" always opens the file picker, also when a photo is already shown.
    UI.tabUpload.addEventListener('click', async () => {
      await switchSource('upload');
      UI.fileInput.click();
    });
    UI.tabCapture.addEventListener('click', () => switchSource('capture'));
    UI.btnCapture.addEventListener('click', captureStill);
    UI.btnCaptureCancel.addEventListener('click', () => switchSource('upload'));

    UI.dropzone.addEventListener('click', () => UI.fileInput.click());
    UI.fileInput.addEventListener('change', handleFileSelect);
    // Drop an image anywhere on the viewer, at any time.
    UI.viewportArea.addEventListener('dragover', (e) => { e.preventDefault(); UI.viewportArea.classList.add('fcf-drop-hover'); });
    UI.viewportArea.addEventListener('dragleave', () => UI.viewportArea.classList.remove('fcf-drop-hover'));
    UI.viewportArea.addEventListener('drop', async (e) => {
      e.preventDefault();
      UI.viewportArea.classList.remove('fcf-drop-hover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        await switchSource('upload');
        processUploadedFile(e.dataTransfer.files[0]);
      }
    });

    UI.modelSelectBaseline.addEventListener('click', () => selectModel('baseline'));
    UI.modelSelectApac.addEventListener('click', () => selectModel('apac'));
    UI.compareToggle.addEventListener('change', (e) => {
      state.compareMode = e.target.checked;
      toggleCompareMode(state.compareMode);
    });

    const layer = (el, key) => el.addEventListener('change', (e) => { state[key] = e.target.checked; redrawActiveCanvas(); });
    layer(UI.chkShowLandmarks, 'showLandmarks');
    layer(UI.chkShowMesh, 'showConnections');
    layer(UI.chkShowPredictions, 'showPredictions');
    layer(UI.chkShowGroundTruth, 'showGroundTruth');
    layer(UI.chkShowErrorVectors, 'showErrorVectors');

    UI.btnZoomIn.addEventListener('click', () => { state.zoomLevel = Math.min(state.zoomLevel + 0.25, 3.5); redrawActiveCanvas(); });
    UI.btnZoomOut.addEventListener('click', () => { state.zoomLevel = Math.max(state.zoomLevel - 0.25, 0.75); redrawActiveCanvas(); });
    UI.btnResetZoom.addEventListener('click', () => { state.zoomLevel = 1.0; state.panX = 0; state.panY = 0; redrawActiveCanvas(); });

    UI.coordsSearchInput.addEventListener('input', (e) => filterCoordinatesTable(e.target.value.toLowerCase()));
    UI.btnExportCsv.addEventListener('click', exportCoordinatesToCsv);
    if (UI.annotationFileInput) UI.annotationFileInput.addEventListener('change', handleCustomAnnotationUpload);

    setupGalleryControls();
  }

  // ------------------------------------------------------------------ sources

  async function switchSource(sourceType) {
    if (state.activeSource === sourceType) return;
    stopWebcam();
    stopCapture();
    state.activeSource = sourceType;
    UI.tabWebcam.classList.toggle('active', sourceType === 'webcam');
    UI.tabUpload.classList.toggle('active', sourceType === 'upload');
    UI.tabCapture.classList.toggle('active', sourceType === 'capture');
    if (sourceType === 'webcam') {
      UI.dropzone.style.display = 'none';
      await startWebcam();
    } else if (sourceType === 'capture') {
      UI.dropzone.style.display = 'none';
      await startCapture();
    } else {
      if (state.currentImageElement) {
        UI.dropzone.style.display = 'none';
        runDetectionOnImage(state.currentImageElement);
      } else {
        UI.dropzone.style.display = 'flex';
      }
    }
  }

  async function startWebcam() {
    try {
      UI.metricFaceStatus.textContent = "Requesting Camera...";
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false
      });
      state.webcamStream = stream;
      UI.videoEl.srcObject = stream;
      await UI.videoEl.play();
      UI.metricFaceStatus.textContent = "Camera Active";
      state.currentVideoElement = UI.videoEl;
      state.currentGroundTruth = null;
      state.metrics = { baseline: null, apac: null };
      state.apacDetector.resetTracking();
      updateGroundTruthDisplay();
      startWebcamLoop();
    } catch (err) {
      console.error("Webcam access error:", err);
      UI.metricFaceStatus.textContent = "Camera Unavailable";
      alert("Unable to access webcam. Please check browser permissions or use Upload Photo mode.");
      switchSource('upload');
    }
  }

  function stopWebcam() {
    if (state.animFrameId) {
      cancelAnimationFrame(state.animFrameId);
      state.animFrameId = null;
    }
    if (state.webcamStream) {
      state.webcamStream.getTracks().forEach(track => track.stop());
      state.webcamStream = null;
    }
    state.isProcessingFrame = false;
    if (state.apacDetector) state.apacDetector.resetTracking();
  }

  /**
   * Live loop. MediaPipe runs whenever it is displayed or APAC-Face needs a face location;
   * APAC-Face otherwise tracks the face from its previous landmarks.
   */
  function startWebcamLoop() {
    async function loop() {
      if (state.activeSource !== 'webcam') return;
      if (!state.isProcessingFrame && UI.videoEl.readyState >= 2) {
        state.isProcessingFrame = true;
        try {
          const wantBaseline = state.compareMode || state.activeModelKey === 'baseline';
          const wantApac = state.compareMode || state.activeModelKey === 'apac';
          const needSeed = wantApac && state.apacDetector.needsSeed();
          let mp = null;
          if (wantBaseline || needSeed) {
            mp = await state.baselineDetector.detect(UI.videoEl);
            state.results.baseline = mp;
          }
          if (wantApac) {
            state.results.apac = await state.apacDetector.detect(UI.videoEl, {
              mediapipeLandmarks: mp && mp.faceDetected ? mp.landmarks : null,
              seedChecked: Boolean(mp)
            });
          }
          if (state.compareMode) {
            renderCanvas(UI.canvasCompareBaseline, UI.videoEl, landmarksOf('baseline'), null, null, 'baseline');
            renderCanvas(UI.canvasCompareApac, UI.videoEl, landmarksOf('apac'), null, null, 'apac');
          } else {
            renderCanvas(UI.mainCanvas, UI.videoEl, landmarksOf(state.activeModelKey), null, null, state.activeModelKey);
          }
          const active = state.results[state.activeModelKey];
          if (active) updateLiveMetrics(active);
          state.frameCount++;
          if (state.frameCount % 10 === 0) {
            updateGroundTruthDisplay();
            if (active) updateCoordinatesTable(active.landmarks);
          }
        } catch (e) {
          console.warn("Frame processing exception:", e);
        } finally {
          state.isProcessingFrame = false;
        }
      }
      state.animFrameId = requestAnimationFrame(loop);
    }
    state.animFrameId = requestAnimationFrame(loop);
  }

  // ------------------------------------------------------------------ Take Photo (camera still)

  async function startCapture() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false
      });
      state.webcamStream = stream;
      UI.videoEl.srcObject = stream;
      await UI.videoEl.play();
      UI.stageSingle.style.display = 'flex';   // the preview always uses the single viewer
      UI.stageCompare.style.display = 'none';
      UI.captureBar.classList.add('active');
      UI.metricFaceStatus.textContent = "Camera ready - press Capture Photo";
      const preview = () => {
        if (state.activeSource !== 'capture') return;
        renderCanvas(UI.mainCanvas, UI.videoEl, [], null, null, state.activeModelKey);
        state.animFrameId = requestAnimationFrame(preview);
      };
      preview();
    } catch (err) {
      console.error("Camera access error:", err);
      alert("Unable to access the camera. Check browser permissions, or use Upload Photo instead.");
      switchSource('upload');
    }
  }

  function stopCapture() {
    if (UI.captureBar) UI.captureBar.classList.remove('active');
  }

  /** Freeze the current camera frame and analyse it exactly like an uploaded photo. */
  function captureStill() {
    const v = UI.videoEl;
    if (!v.videoWidth) return;
    const c = document.createElement('canvas');
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext('2d').drawImage(v, 0, 0);
    const img = new Image();
    img.onload = async () => {
      state.currentImageElement = null;       // don't re-run the previous photo
      await switchSource('upload');           // stops the camera
      toggleCompareMode(state.compareMode);
      showImage(img);
    };
    img.src = c.toDataURL('image/jpeg', 0.95);
  }

  function handleFileSelect(e) {
    if (e.target.files && e.target.files[0]) processUploadedFile(e.target.files[0]);
    e.target.value = '';  // allow choosing the same file again
  }

  /** Show a new photo (upload or capture) and run both models; it has no ground truth. */
  function showImage(img) {
    state.currentImageElement = img;
    UI.imgEl.src = img.src;
    UI.dropzone.style.display = 'none';
    state.currentGroundTruth = null;
    state.currentSampleKey = null;
    runDetectionOnImage(img);
  }

  function processUploadedFile(file) {
    if (!file.type.match('image.*')) {
      alert("Please upload a valid image file (JPG, JPEG, PNG).");
      return;
    }
    const reader = new FileReader();
    reader.onload = function (event) {
      const img = new Image();
      img.onload = () => showImage(img);   // an upload has no ground truth until an annotation file is loaded
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  }

  function landmarksOf(key) {
    const r = state.results[key];
    return r && r.faceDetected ? r.landmarks : [];
  }

  /** Runs BOTH models on an image, then computes ground-truth metrics for each when GT exists. */
  async function runDetectionOnImage(imgElement) {
    if (!imgElement) return;
    const mp = await state.baselineDetector.detect(imgElement);
    const apac = await state.apacDetector.detect(imgElement, { mediapipeLandmarks: mp.faceDetected ? mp.landmarks : null });
    if (imgElement !== state.currentImageElement) return;  // a newer photo was loaded meanwhile
    state.results = { baseline: mp, apac };

    const gt = state.currentGroundTruth && state.currentGroundTruth.landmarks;
    state.metrics = {
      baseline: gt ? state.datasetManager.calculateMetrics(landmarksOf('baseline'), gt) : null,
      apac: gt ? state.datasetManager.calculateMetrics(landmarksOf('apac'), gt) : null
    };
    updateLiveMetrics(state.results[state.activeModelKey]);
    updateGroundTruthDisplay();
    redrawActiveCanvas();
    updateCoordinatesTable(landmarksOf(state.activeModelKey));
  }

  function redrawActiveCanvas() {
    const media = state.activeSource === 'webcam' ? state.currentVideoElement : state.currentImageElement;
    if (!media) return;
    const gt = state.currentGroundTruth ? state.currentGroundTruth.landmarks : null;
    const errs = (k) => (state.metrics[k] ? state.metrics[k].pointErrors : null);
    if (state.compareMode) {
      renderCanvas(UI.canvasCompareBaseline, media, landmarksOf('baseline'), gt, errs('baseline'), 'baseline');
      renderCanvas(UI.canvasCompareApac, media, landmarksOf('apac'), gt, errs('apac'), 'apac');
    } else {
      const k = state.activeModelKey;
      renderCanvas(UI.mainCanvas, media, landmarksOf(k), gt, errs(k), k);
    }
  }

  /**
   * High-DPI canvas renderer: photo, mesh, error vectors (prediction -> ground truth),
   * ground truth (emerald) and predictions (MediaPipe cyan, APAC-Face amber).
   */
  function renderCanvas(canvas, mediaElement, predictedLandmarks, groundTruthLandmarks, errorVectors, modelKey) {
    if (!canvas || !mediaElement) return;
    const ctx = canvas.getContext('2d');
    const width = mediaElement.videoWidth || mediaElement.naturalWidth || mediaElement.width || 448;
    const height = mediaElement.videoHeight || mediaElement.naturalHeight || mediaElement.height || 448;
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const scale = Math.max(1, Math.min(width, height) / 448);

    ctx.save();
    ctx.clearRect(0, 0, width, height);
    if (state.zoomLevel !== 1.0 || state.panX !== 0 || state.panY !== 0) {
      ctx.translate(width / 2, height / 2);
      ctx.scale(state.zoomLevel, state.zoomLevel);
      ctx.translate(-width / 2 + state.panX, -height / 2 + state.panY);
    }
    ctx.drawImage(mediaElement, 0, 0, width, height);

    const isApac = modelKey === 'apac';
    if (state.showConnections && state.showPredictions && predictedLandmarks && predictedLandmarks.length > 0) {
      const connections = isApac ? state.apacDetector.getConnections() : state.baselineDetector.getConnections();
      ctx.strokeStyle = isApac ? "rgba(245, 158, 11, 0.55)" : "rgba(0, 229, 255, 0.22)";
      ctx.lineWidth = (isApac ? 1.4 : 0.8) * scale;
      ctx.beginPath();
      for (const [a, b] of connections) {
        const p1 = predictedLandmarks[a];
        const p2 = predictedLandmarks[b];
        if (p1 && p2) {
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
        }
      }
      ctx.stroke();
    }

    if (state.showErrorVectors && errorVectors && errorVectors.length > 0) {
      ctx.strokeStyle = "rgba(239, 68, 68, 0.85)";
      ctx.lineWidth = 1.5 * scale;
      ctx.fillStyle = "#EF4444";
      for (const err of errorVectors) {
        ctx.beginPath();
        ctx.moveTo(err.predX, err.predY);
        ctx.lineTo(err.gtX, err.gtY);
        ctx.stroke();
      }
    }

    if (state.showGroundTruth && groundTruthLandmarks && groundTruthLandmarks.length > 0) {
      ctx.fillStyle = "#10B981";
      ctx.shadowColor = "rgba(16, 185, 129, 0.6)";
      ctx.shadowBlur = 6;
      for (const pt of groundTruthLandmarks) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, 2.6 * scale, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    }

    if (state.showLandmarks && state.showPredictions && predictedLandmarks && predictedLandmarks.length > 0) {
      ctx.fillStyle = isApac ? "#F59E0B" : "#00E5FF";
      ctx.shadowColor = isApac ? "rgba(245, 158, 11, 0.5)" : "rgba(0, 229, 255, 0.5)";
      ctx.shadowBlur = 4;
      const radius = (predictedLandmarks.length > 100 ? 1.6 : 2.6) * scale;
      for (const pt of predictedLandmarks) {
        ctx.beginPath();
        ctx.arc(pt.x, pt.y, radius, 0, 2 * Math.PI);
        ctx.fill();
      }
      ctx.shadowBlur = 0;
    }
    ctx.restore();
  }

  function toggleCompareMode(isCompare) {
    UI.stageSingle.style.display = isCompare ? 'none' : 'flex';
    UI.stageCompare.style.display = isCompare ? 'grid' : 'none';
    redrawActiveCanvas();
  }

  function selectModel(modelKey) {
    state.activeModelKey = modelKey;
    UI.modelSelectBaseline.classList.toggle('active', modelKey === 'baseline');
    UI.modelSelectApac.classList.toggle('active', modelKey === 'apac');
    const r = state.results[modelKey];
    if (r) updateLiveMetrics(r);
    updateGroundTruthDisplay();
    redrawActiveCanvas();
    updateCoordinatesTable(landmarksOf(modelKey));
  }

  // ------------------------------------------------------------------ panels

  function updateLiveMetrics(result) {
    if (!result) return;
    const n = result.faceDetected ? result.landmarks.length : 0;
    UI.metricLandmarksCount.textContent = `${n} pts`;
    UI.metricFaceStatus.textContent = result.error ? "Model unavailable"
      : (result.trackState === "fallback" ? "Centred-crop fallback" : (result.faceDetected ? "Face Detected" : "No Face Found"));
    UI.metricProcessingTime.textContent = `${result.executionTimeMs || 0} ms`;
    UI.metricStability.textContent = result.stability || "—";
    UI.metricModelName.textContent = result.modelName || MODEL_LABEL[state.activeModelKey];
  }

  const fmt = (v, unit) => (v === null || v === undefined || Number.isNaN(v)) ? "—" : `${v}${unit}`;

  /** The four ground-truth cards (active model) plus the side-by-side comparison of both models. */
  function updateGroundTruthDisplay() {
    const key = state.activeModelKey;
    const m = state.metrics[key];
    if (UI.gtModelLabel) UI.gtModelLabel.textContent = MODEL_LABEL[key];

    if (!m) {
      const why = state.activeSource === 'webcam'
        ? "Live webcam frames have no ground-truth annotations, so no error can be measured on them. Below: the errors both models made on the 2,500-face WFLW benchmark, and how far apart the two models are right now."
        : (state.currentGroundTruth
           ? "No face was found by this model, so its error cannot be computed."
           : "This photo has no ground-truth annotations. Load a WFLW benchmark face below (real manual annotations) or upload an annotation file for your photo.");
      UI.gtStatusNotice.textContent = why;
      UI.gtStatusNotice.className = "fcf-badge fcf-badge-amber";
      UI.gtMeanError.textContent = "—";
      UI.gtNme.textContent = "—";
      UI.gtMaxError.textContent = "—";
      UI.gtFailureRate.textContent = "—";
      UI.gtPointsCompared.textContent = "0";
      UI.gtFormulaDetails.textContent = "NME = (1/N) × Σ (||p_i − g_i||₂ / d_ocular) × 100%. Requires real ground-truth coordinates.";
    } else {
      UI.gtStatusNotice.textContent = `Ground truth: ${state.currentGroundTruth.source}. ${m.pointsCompared} points compared (68-point layout).`;
      UI.gtStatusNotice.className = "fcf-badge fcf-badge-emerald";
      UI.gtMeanError.textContent = `${m.meanErrorPx} px`;
      UI.gtNme.textContent = `${m.nmePercent}%`;
      UI.gtMaxError.textContent = `${m.maxErrorPx} px`;
      UI.gtFailureRate.textContent = `${m.failureRatePercent}%`;
      UI.gtPointsCompared.textContent = `${m.pointsCompared}`;
      UI.gtFormulaDetails.textContent = `Inter-ocular distance (outer eye corners): ${m.iodPx} px. Failure: a point whose error exceeds 8% of it. Computed from the actual predicted and annotated coordinates.`;
    }
    updateComparisonPanel();
  }

  function updateComparisonPanel() {
    if (!UI.gtCompareBody) return;
    const mb = state.metrics.baseline;
    const ma = state.metrics.apac;
    const rows = [
      ["Mean Euclidean Error", "meanErrorPx", " px"],
      ["Normalized Mean Error (NME)", "nmePercent", "%"],
      ["Maximum Landmark Error", "maxErrorPx", " px"],
      ["Failure Rate (>8% NME)", "failureRatePercent", "%"]
    ];
    if (mb || ma) {
      UI.gtCompareBody.innerHTML = rows.map(([label, k, unit]) => {
        const b = mb ? mb[k] : null;
        const a = ma ? ma[k] : null;
        const better = (a !== null && b !== null) ? (a < b ? 'apac' : (b < a ? 'baseline' : null)) : null;
        return `<tr><td>${label}</td>
          <td class="${better === 'baseline' ? 'fcf-win' : ''}">${fmt(b, unit)}</td>
          <td class="${better === 'apac' ? 'fcf-win' : ''}">${fmt(a, unit)}</td></tr>`;
      }).join('');
      if (mb && ma) {
        const red = ((mb.nmePercent - ma.nmePercent) / mb.nmePercent * 100);
        UI.gtCompareSummary.textContent = red >= 0
          ? `On this face APAC-Face's NME is ${red.toFixed(1)}% lower than MediaPipe's (${ma.nmePercent}% vs ${mb.nmePercent}%).`
          : `On this face MediaPipe's NME is lower (${mb.nmePercent}% vs ${ma.nmePercent}%). Single faces vary - see the 2,500-face benchmark below.`;
      } else {
        UI.gtCompareSummary.textContent = `${mb ? MODEL_LABEL.apac : MODEL_LABEL.baseline} found no face on this image.`;
      }
      const ap = state.results.apac;
      if (ap && ap.trackState === "fallback") {
        UI.gtCompareSummary.textContent += " MediaPipe gave no face location, so APAC-Face ran on a centred crop with two refinement passes.";
      }
      return;
    }

    // No ground truth: reference errors from the benchmark plus live inter-model agreement.
    const bm = state.benchmark && state.benchmark.mediapipe;
    const agree = state.datasetManager && state.datasetManager.calculateAgreement(landmarksOf('baseline'), landmarksOf('apac'));
    if (bm) {
      const B = bm.methods.mediapipe.detected_faces;
      const web = state.benchmark.web_model || 'proposed';
      const A = (bm.methods[`${web}_mp_located`] || bm.methods[web]).detected_faces;
      UI.gtCompareBody.innerHTML = `
        <tr><td>Mean Euclidean Error (benchmark)</td><td>${B.mean_error_px.toFixed(2)} px</td><td class="fcf-win">${A.mean_error_px.toFixed(2)} px</td></tr>
        <tr><td>NME, 68 points (benchmark)</td><td>${(B.nme * 100).toFixed(2)}%</td><td class="fcf-win">${(A.nme * 100).toFixed(2)}%</td></tr>
        <tr><td>Point failure rate &gt;8% (benchmark)</td><td>${(B.point_failure_rate * 100).toFixed(1)}%</td><td class="fcf-win">${(A.point_failure_rate * 100).toFixed(1)}%</td></tr>
        <tr><td>Face failure rate, NME &gt;10% (benchmark)</td><td>${(B.failure_rate * 100).toFixed(1)}%</td><td class="fcf-win">${(A.failure_rate * 100).toFixed(1)}%</td></tr>`;
      UI.gtCompareSummary.textContent = `Benchmark values: WFLW test, ${bm.faces_detected.toLocaleString()} faces with manual annotations. ` +
        (agree ? `On the current image the two models' points are ${agree.meanPx} px apart on average (${agree.percentOfIod}% of the inter-ocular distance) - disagreement, not error.` : "");
    } else {
      UI.gtCompareBody.innerHTML = `<tr><td colspan="3" style="text-align:center;">No ground truth for this view.</td></tr>`;
      UI.gtCompareSummary.textContent = "";
    }
  }

  // ------------------------------------------------------------------ benchmark results section

  async function loadBenchmarkResults() {
    try {
      const resp = await fetch('./data/benchmark_results.json');
      if (resp.ok) state.benchmark = await resp.json();
    } catch (e) { /* rendered as missing below */ }
    renderBenchmarkResults();
  }

  function renderBenchmarkResults() {
    const kpis = document.getElementById('fcf-bench-kpis');
    const chart = document.getElementById('fcf-bench-chart');
    const subs = document.getElementById('fcf-bench-subsets-tbody');
    const regs = document.getElementById('fcf-bench-regions-tbody');
    const apacBody = document.getElementById('fcf-bench-apac-tbody');
    const note = document.getElementById('fcf-bench-note');
    const bm = state.benchmark && state.benchmark.mediapipe;
    if (!kpis) return;
    if (!bm) {
      kpis.innerHTML = `<div class="fcf-gt-card" style="grid-column:1/-1;">Benchmark results not found. Run <code>python -m src.mediapipe_baseline</code> and <code>python -m src.export_web</code>.</div>`;
      return;
    }
    const M = bm.methods;
    const web = state.benchmark.web_model || 'proposed';        // the model this page runs
    const mp = M.mediapipe.detected_faces;
    const mc = M.mediapipe_calibrated.detected_faces;
    const std = M[web].detected_faces;                           // standard WFLW protocol (ground-truth crop)
    const locKey = M[`${web}_mp_located`] ? `${web}_mp_located` : web;
    const ap = M[locKey].detected_faces;                         // same face location as MediaPipe (what this page runs)
    const cmp = bm.comparison[`${locKey}_vs_mediapipe`];
    const cmpCal = bm.comparison[`${locKey}_vs_mediapipe_calibrated`];
    const pct = (v, d = 2) => `${(v * 100).toFixed(d)}%`;

    kpis.innerHTML = `
      <div class="fcf-gt-card"><div class="fcf-metric-label">NME · Google MediaPipe</div>
        <div class="fcf-metric-val">${pct(mp.nme)}</div><div class="fcf-metric-sub">68 points, ${bm.faces_detected.toLocaleString()} faces</div></div>
      <div class="fcf-gt-card fcf-gt-card-emerald"><div class="fcf-metric-label">NME · APAC-Face</div>
        <div class="fcf-metric-val" style="color: var(--faceify-amber);">${pct(ap.nme)}</div><div class="fcf-metric-sub">same faces, same face location as MediaPipe</div></div>
      <div class="fcf-gt-card"><div class="fcf-metric-label">Error reduction vs MediaPipe</div>
        <div class="fcf-metric-val" style="color: var(--faceify-emerald);">−${cmp.nme_reduction_pct.toFixed(1)}%</div>
        <div class="fcf-metric-sub">95% CI of NME drop ${(cmp.bootstrap_nme_diff.ci_low * 100).toFixed(2)}–${(cmp.bootstrap_nme_diff.ci_high * 100).toFixed(2)} pts</div></div>
      <div class="fcf-gt-card"><div class="fcf-metric-label">Mean Euclidean Error</div>
        <div class="fcf-metric-val">${ap.mean_error_px.toFixed(2)} px</div><div class="fcf-metric-sub">MediaPipe ${mp.mean_error_px.toFixed(2)} px</div></div>`;

    const bars = [
      ["Google MediaPipe", mp.nme, "fcf-bar-mp"],
      ["MediaPipe, offset-corrected", mc.nme, "fcf-bar-mp-cal"],
      ["APAC-Face, same face location", ap.nme, "fcf-bar-apac"],
      ["APAC-Face, standard WFLW crop", std.nme, "fcf-bar-apac-std"]
    ];
    const max = Math.max(...bars.map(b => b[1]));
    chart.innerHTML = bars.map(([label, v, cls]) => `
      <div class="fcf-bar-row" title="${label}: ${pct(v)} NME">
        <div class="fcf-bar-label">${label}</div>
        <div class="fcf-bar-track"><div class="fcf-bar ${cls}" style="width:${(v / max * 100).toFixed(1)}%"></div></div>
        <div class="fcf-bar-value">${pct(v)}</div>
      </div>`).join('');

    const COND = { pose: "Large pose", expression: "Exaggerated expression", illumination: "Extreme illumination",
                   makeup: "Make-up", occlusion: "Occlusion", blur: "Blur" };
    subs.innerHTML = Object.entries(COND).map(([k, label]) => {
      const a = bm.subsets[locKey][k], b = bm.subsets.mediapipe[k], c = bm.subsets.mediapipe_calibrated[k];
      return `<tr><td>${label}</td><td>${a.faces}</td><td>${pct(b.nme)}</td><td>${pct(c.nme)}</td>
              <td class="fcf-win">${pct(a.nme)}</td></tr>`;
    }).join('');

    regs.innerHTML = Object.keys(ap.regions).map(r => {
      const vals = [mp.regions[r], mc.regions[r], ap.regions[r]];
      const best = vals.indexOf(Math.min(...vals));
      return `<tr><td>${r}</td>${vals.map((v, i) => `<td class="${i === best ? 'fcf-win' : ''}">${pct(v)}</td>`).join('')}</tr>`;
    }).join('');

    const ac = state.benchmark.apac_consistency;
    if (ac && apacBody) {
      apacBody.innerHTML = Object.values(ac.groups).map(g => `
        <tr><td>${g.label}${g.apac ? '' : ' <span style="color:var(--faceify-text-dim)">(reference)</span>'}</td><td>${g.images}</td>
        <td>${pct(g.models.baseline.flip)}</td><td class="fcf-win">${pct(g.models.proposed.flip)}</td>
        <td>${pct(g.models.baseline.crop)}</td><td class="fcf-win">${pct(g.models.proposed.crop)}</td></tr>`).join('');
    }

    const innerSig = cmpCal.bootstrap_inner_nme_diff.p_value < 0.05;
    note.innerHTML = `
      <strong>How this was measured.</strong> ${bm.protocol.dataset}; ${bm.protocol.layout}; normalised by ${bm.protocol.normalisation}.
      MediaPipe ${bm.mediapipe_version} detected ${pct(bm.detection_rate, 1)} of the faces; every number uses those faces for every method.
      Headline APAC-Face numbers use the <em>same face location as MediaPipe</em> (our crop is built from MediaPipe's own detection,
      exactly as on this page); the standard WFLW protocol, which crops around the annotated face, gives ${pct(std.nme)}.
      "Offset-corrected" removes MediaPipe's systematic landmark-convention offset (learned on WFLW training faces) so it is judged on
      localisation, not labelling style. Against it APAC-Face still cuts 68-point NME by ${cmpCal.nme_reduction_pct.toFixed(1)}%
      (CI ${(cmpCal.bootstrap_nme_diff.ci_low * 100).toFixed(2)}–${(cmpCal.bootstrap_nme_diff.ci_high * 100).toFixed(2)} pts);
      on the inner face points alone (without the jaw line) the difference is ${cmpCal.inner_nme_reduction_pct.toFixed(1)}%
      ${innerSig ? `(significant, p = ${cmpCal.bootstrap_inner_nme_diff.p_value.toFixed(3)})` : `and <strong>not statistically significant</strong> (p = ${cmpCal.bootstrap_inner_nme_diff.p_value.toFixed(2)})`}.
      APAC-Face was trained on WFLW's annotation convention, which favours it on this test set.`;
  }

  // ------------------------------------------------------------------ gallery

  function updateGalleryTabCounts() {
    const counts = state.datasetManager.categoryCounts();
    const labels = { all: "All Faces", gt: "WFLW · Ground Truth", apac: "APAC Inclusive", "East Asia": "East Asia",
                     "South Asia": "South Asia / Indian", "Southeast Asia": "Southeast Asia", other: "Other Demographics" };
    document.querySelectorAll('#fcf-gallery-filter-tabs [data-filter]').forEach(btn => {
      const f = btn.getAttribute('data-filter');
      btn.textContent = `${labels[f] || f} (${(counts[f] || 0).toLocaleString()})`;
    });
    const notes = document.getElementById('fcf-gallery-load-notes');
    if (notes) notes.textContent = state.datasetManager.loadNotes.join(' ');
  }

  function renderBenchmarkGallery() {
    const dm = state.datasetManager;
    const result = dm.getFilteredSamples(state.galleryFilter, state.galleryQuery, state.galleryPage, state.galleryPageSize);
    state.galleryPage = result.currentPage;
    UI.galleryContainer.innerHTML = '';

    if (result.items.length === 0) {
      UI.galleryContainer.innerHTML = `
        <div style="grid-column: 1/-1; text-align: center; padding: 48px; color: var(--faceify-text-muted);">
          No samples for this filter. ${dm.loadNotes.join(' ')}
        </div>`;
    }
    for (const sample of result.items) {
      const region = sample.region || 'APAC Diverse';
      const isAPAC = ['East Asia', 'South Asia', 'Southeast Asia'].includes(region);
      const card = document.createElement('div');
      card.className = 'fcf-gallery-card';
      const tag = sample.hasGroundTruth ? (sample.conditions || []).join(' · ') : region;
      card.innerHTML = `
        <div class="fcf-gallery-thumb-wrap">
          <img src="${sample.imagePath}" alt="${sample.name}" class="fcf-gallery-img" loading="lazy">
          <div class="fcf-gallery-tag" style="background:${sample.hasGroundTruth ? 'rgba(16,185,129,0.25)' : (isAPAC ? 'rgba(0,229,255,0.15)' : 'rgba(0,23,56,0.85)')};">${tag}</div>
        </div>
        <div class="fcf-gallery-body">
          <div class="fcf-gallery-title" style="font-size:0.95rem;">${sample.name}</div>
          <div class="fcf-gallery-meta">${sample.hasGroundTruth ? 'Manual 98-point annotation' : 'No landmark annotation'}</div>
          <div class="fcf-gallery-metrics-row">
            <span><strong>ID:</strong> ${sample.id}</span>
            <span class="${sample.hasGroundTruth ? 'fcf-emerald-text' : (isAPAC ? 'fcf-cyan-text' : '')}">${sample.hasGroundTruth ? '✓ Ground truth' : (isAPAC ? '✓ APAC' : 'Diverse')}</span>
          </div>
          <button class="fcf-btn fcf-btn-primary fcf-btn-sm" style="width:100%;">
            ${sample.hasGroundTruth ? 'Load &amp; Measure Error' : 'Load &amp; Detect'}
          </button>
        </div>`;
      card.querySelector('button').addEventListener('click', () => loadBenchmarkSample(sample.key, sample.imagePath));
      UI.galleryContainer.appendChild(card);
    }

    const showingText = `Showing ${result.startIndex}–${result.endIndex} of ${result.totalCount.toLocaleString()} samples`;
    const statusEl = document.getElementById('fcf-gallery-status-count');
    const pageTopEl = document.getElementById('fcf-gallery-page-indicator');
    const pageBotEl = document.getElementById('fcf-gallery-page-indicator-bottom');
    if (statusEl) statusEl.textContent = showingText;
    if (pageTopEl) pageTopEl.textContent = `${result.currentPage} / ${result.totalPages.toLocaleString()}`;
    if (pageBotEl) pageBotEl.textContent = `Page ${result.currentPage} of ${result.totalPages.toLocaleString()}`;
  }

  function setupGalleryControls() {
    const tabContainer = document.getElementById('fcf-gallery-filter-tabs');
    if (tabContainer) {
      tabContainer.querySelectorAll('[data-filter]').forEach(btn => {
        btn.addEventListener('click', () => {
          tabContainer.querySelectorAll('[data-filter]').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          state.galleryFilter = btn.getAttribute('data-filter');
          state.galleryPage = 1;
          renderBenchmarkGallery();
        });
      });
    }
    const searchEl = document.getElementById('fcf-gallery-search');
    if (searchEl) {
      let debounceTimer;
      searchEl.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          state.galleryQuery = e.target.value;
          state.galleryPage = 1;
          renderBenchmarkGallery();
        }, 300);
      });
    }
    const go = (delta) => {
      state.galleryPage = Math.max(1, state.galleryPage + delta);
      renderBenchmarkGallery();
      document.getElementById('benchmark-gallery').scrollIntoView({ behavior: 'smooth' });
    };
    ['fcf-gallery-prev-btn', 'fcf-gallery-prev-btn-bottom'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('click', () => go(-1));
    });
    ['fcf-gallery-next-btn', 'fcf-gallery-next-btn-bottom'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('click', () => go(1));
    });
  }

  async function loadBenchmarkSample(sampleKey, imagePath, scroll = true) {
    state.currentSampleKey = sampleKey;
    if (state.activeSource !== 'upload') {
      state.currentImageElement = null;   // nothing to re-run: the sample loads next
      await switchSource('upload');
      toggleCompareMode(state.compareMode);
    }
    const img = new Image();
    img.onload = async function () {
      state.currentImageElement = img;
      UI.imgEl.src = img.src;
      UI.dropzone.style.display = 'none';
      state.currentGroundTruth = await state.datasetManager.getGroundTruth(sampleKey);  // null if none
      await runDetectionOnImage(img);
      if (scroll) document.getElementById('fcf-landmark-studio').scrollIntoView({ behavior: 'smooth' });
    };
    img.onerror = () => console.warn(`Image not found: ${imagePath}`);
    img.src = imagePath;
  }

  function handleCustomAnnotationUpload(e) {
    if (!e.target.files || !e.target.files[0]) return;
    const file = e.target.files[0];
    const reader = new FileReader();
    reader.onload = function (event) {
      const content = event.target.result;
      let parsed = null;
      try {
        if (file.name.endsWith('.json')) parsed = state.datasetManager.parseJson(content);
        else if (file.name.endsWith('.txt') || file.name.endsWith('.pts')) parsed = state.datasetManager.parsePtsTxt(content);
        else if (file.name.endsWith('.csv')) parsed = state.datasetManager.parseCsv(content);
      } catch (err) {
        parsed = null;
      }
      if (parsed && parsed.landmarks && parsed.landmarks.length === 68) {
        parsed.source = `Uploaded annotation (${file.name})`;
        state.currentGroundTruth = parsed;
        alert(`Loaded ${parsed.landmarks.length} ground-truth points from ${file.name}.`);
        if (state.currentImageElement) runDetectionOnImage(state.currentImageElement);
      } else {
        alert("The annotation must contain 68 points (iBUG layout) in pixel coordinates of the loaded image.");
      }
    };
    reader.readAsText(file);
  }

  // ------------------------------------------------------------------ coordinates

  function updateCoordinatesTable(landmarks) {
    if (!landmarks || landmarks.length === 0) {
      UI.coordsTableBody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--faceify-text-muted);">No landmarks detected yet. Enable webcam or select a benchmark sample.</td></tr>`;
      UI.coordsCountLabel.textContent = "0 points";
      return;
    }
    UI.coordsCountLabel.textContent = `${landmarks.length} points · ${MODEL_LABEL[state.activeModelKey]}`;
    const displayCount = Math.min(landmarks.length, 120);
    let html = '';
    for (let i = 0; i < displayCount; i++) {
      const pt = landmarks[i];
      const region = state.datasetManager.getRegionForLayout(i, landmarks.length);
      html += `
        <tr data-point-id="${pt.id}" data-region="${region.toLowerCase()}">
          <td>#${pt.id}</td><td>${pt.x} px</td><td>${pt.y} px</td>
          <td>${pt.normalized_x}</td><td>${pt.normalized_y}</td>
          <td><span class="fcf-badge" style="margin:0; padding:2px 8px; font-size:0.75rem;">${region}</span></td>
        </tr>`;
    }
    if (landmarks.length > displayCount) {
      html += `<tr><td colspan="6" style="text-align:center; color: var(--faceify-text-muted); padding: 12px;">... and ${landmarks.length - displayCount} more points (Export CSV for all) ...</td></tr>`;
    }
    UI.coordsTableBody.innerHTML = html;
  }

  function filterCoordinatesTable(query) {
    UI.coordsTableBody.querySelectorAll('tr[data-point-id]').forEach(row => {
      const id = row.getAttribute('data-point-id');
      const region = row.getAttribute('data-region') || '';
      row.style.display = (!query || id.includes(query) || region.includes(query)) ? '' : 'none';
    });
  }

  function exportCoordinatesToCsv() {
    const landmarks = landmarksOf(state.activeModelKey);
    if (!landmarks.length) {
      alert("No landmark coordinates available to export.");
      return;
    }
    let csv = "point_id,x_px,y_px,normalized_x,normalized_y,region\n";
    landmarks.forEach((pt, idx) => {
      csv += `${pt.id},${pt.x},${pt.y},${pt.normalized_x},${pt.normalized_y},"${state.datasetManager.getRegionForLayout(idx, landmarks.length)}"\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.setAttribute("href", url);
    link.setAttribute("download", `faceify_landmarks_${state.activeModelKey}_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  // ------------------------------------------------------------------ hero animation (unchanged)

  function setupHeroCanvas() {
    const canvas = document.getElementById('fcf-hero-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = 480;
    canvas.height = 480;
    const points = [];
    for (let i = 0; i < 85; i++) {
      const theta = Math.random() * 2.0 * Math.PI;
      const phi = Math.acos(2.0 * Math.random() - 1.0);
      const r = 160;
      points.push({ x: r * Math.sin(phi) * Math.cos(theta), y: r * Math.sin(phi) * Math.sin(theta) * 1.25, z: r * Math.cos(phi) });
    }
    let angle = 0;
    function animate() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const cx = canvas.width / 2;
      const cy = canvas.height / 2;
      angle += 0.008;
      const projected = points.map(pt => {
        const rotX = pt.x * Math.cos(angle) - pt.z * Math.sin(angle);
        const rotZ = pt.x * Math.sin(angle) + pt.z * Math.cos(angle);
        const scale = 380 / (380 + rotZ);
        return { x: cx + rotX * scale, y: cy + pt.y * scale, scale };
      });
      ctx.strokeStyle = "rgba(0, 229, 255, 0.15)";
      ctx.lineWidth = 0.8;
      for (let i = 0; i < projected.length; i++) {
        for (let j = i + 1; j < projected.length; j++) {
          if (Math.hypot(projected[i].x - projected[j].x, projected[i].y - projected[j].y) < 48) {
            ctx.beginPath();
            ctx.moveTo(projected[i].x, projected[i].y);
            ctx.lineTo(projected[j].x, projected[j].y);
            ctx.stroke();
          }
        }
      }
      for (const p of projected) {
        ctx.fillStyle = `rgba(0, 229, 255, ${Math.max(0.2, (p.scale - 0.7) * 2)})`;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2.5 * p.scale, 0, 2 * Math.PI);
        ctx.fill();
      }
      requestAnimationFrame(animate);
    }
    animate();
  }

  window.FaceifyApp = { init: initApp, state, switchSource, selectModel, loadBenchmarkSample };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initApp);
  } else {
    initApp();
  }
})();
