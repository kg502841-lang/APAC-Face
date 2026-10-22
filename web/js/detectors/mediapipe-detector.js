/**
 * Faceify AI Facial Landmark Analysis - MediaPipe Baseline Detector
 * -----------------------------------------------------------------
 * Google MediaPipe Face Landmarker (Tasks Vision, 478-vertex face mesh) running 100% in the browser
 * via WebAssembly / WebGL. It loads the same model file (models/face_landmarker.task) that the
 * offline benchmark (`python -m src.mediapipe_baseline`) scores against WFLW ground truth, so the
 * numbers on the page and the live baseline are the same model.
 *
 * If MediaPipe cannot load (offline, blocked CDN) the detector reports the failure instead of
 * inventing landmarks.
 */

const MP_VISION_VERSION = "0.10.14";
const MP_VISION_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VISION_VERSION}`;

class MediaPipeDetector extends LandmarkDetector {
  constructor(options = {}) {
    super("Google MediaPipe (Baseline)", options);
    this.pointCount = 478;
    this.landmarker = null;
    this.runningMode = "IMAGE";
    this.lastTimestamp = 0;
    this.lastFrameLandmarks = null;
    this.connections = [];
    this.delegate = null;
    this.loadError = null;
  }

  async initialize() {
    try {
      const vision = await import(`${MP_VISION_URL}/vision_bundle.mjs`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_VISION_URL}/wasm`);
      for (const delegate of ["GPU", "CPU"]) {
        try {
          this.landmarker = await vision.FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: "./models/face_landmarker.task", delegate },
            runningMode: "IMAGE",
            numFaces: 1,
            minFaceDetectionConfidence: 0.5,
            minFacePresenceConfidence: 0.5,
            minTrackingConfidence: 0.5
          });
          this.delegate = delegate;
          break;
        } catch (e) {
          console.warn(`[MediaPipeDetector] ${delegate} delegate unavailable:`, e);
        }
      }
      if (!this.landmarker) throw new Error("Face Landmarker could not be created");
      this.connections = vision.FaceLandmarker.FACE_LANDMARKS_TESSELATION.map(c => [c.start, c.end]);
      this.isReady = true;
      console.log(`[MediaPipeDetector] Face Landmarker ready (${this.delegate}).`);
    } catch (err) {
      this.loadError = `MediaPipe failed to load: ${err.message || err}`;
      console.error("[MediaPipeDetector]", err);
    }
    return this.isReady;
  }

  async _setMode(mode) {
    if (this.runningMode !== mode) {
      await this.landmarker.setOptions({ runningMode: mode });
      this.runningMode = mode;
    }
  }

  /**
   * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputElement
   */
  async detect(inputElement) {
    const startTime = performance.now();
    const w = inputElement.videoWidth || inputElement.naturalWidth || inputElement.width;
    const h = inputElement.videoHeight || inputElement.naturalHeight || inputElement.height;
    const base = { modelName: this.name, connections: this.connections, isSimulation: false };

    if (!this.landmarker) {
      return { ...base, success: false, faceDetected: false, landmarks: [], pointCount: 0,
               executionTimeMs: 0, stability: "Unavailable", error: this.loadError || "MediaPipe not loaded" };
    }

    let result;
    if (inputElement instanceof HTMLVideoElement) {
      await this._setMode("VIDEO");
      this.lastTimestamp = Math.max(performance.now(), this.lastTimestamp + 1);
      result = this.landmarker.detectForVideo(inputElement, this.lastTimestamp);
    } else {
      await this._setMode("IMAGE");
      result = this.landmarker.detect(inputElement);
      this.lastFrameLandmarks = null;
    }
    const elapsedMs = Math.round(performance.now() - startTime);

    if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
      this.lastFrameLandmarks = null;
      return { ...base, success: true, faceDetected: false, landmarks: [], pointCount: 0,
               executionTimeMs: elapsedMs, stability: "No Face" };
    }

    // MediaPipe coordinates are continuous (0 = image edge); shift to pixel-centre convention,
    // the convention of the WFLW annotations and of the offline benchmark.
    const landmarks = result.faceLandmarks[0].map((pt, idx) => ({
      id: idx,
      x: +(pt.x * w - 0.5).toFixed(2),
      y: +(pt.y * h - 0.5).toFixed(2),
      normalized_x: +pt.x.toFixed(4),
      normalized_y: +pt.y.toFixed(4),
      z: +(pt.z || 0).toFixed(4)
    }));

    return { ...base, success: true, faceDetected: true, landmarks, pointCount: landmarks.length,
             executionTimeMs: elapsedMs, stability: this._calculateStability(landmarks) };
  }

  /**
   * Frame-to-frame landmark displacement (video only). This is stability, NOT accuracy.
   */
  _calculateStability(currentLandmarks) {
    if (!this.lastFrameLandmarks || this.lastFrameLandmarks.length !== currentLandmarks.length) {
      this.lastFrameLandmarks = currentLandmarks;
      return "Single frame";
    }
    const sampleIndices = [0, 4, 33, 61, 133, 159, 145, 263, 291, 362, 386, 374, 152, 10, 234, 454];
    let totalDisp = 0;
    for (const idx of sampleIndices) {
      const dx = currentLandmarks[idx].normalized_x - this.lastFrameLandmarks[idx].normalized_x;
      const dy = currentLandmarks[idx].normalized_y - this.lastFrameLandmarks[idx].normalized_y;
      totalDisp += Math.sqrt(dx * dx + dy * dy);
    }
    this.lastFrameLandmarks = currentLandmarks;
    const avgDisp = totalDisp / sampleIndices.length;
    if (avgDisp < 0.003) return "Excellent (Stable)";
    if (avgDisp < 0.008) return "Good";
    if (avgDisp < 0.02) return "Moderate Motion";
    return "High Motion";
  }

  getConnections() {
    return this.connections;
  }

  getModelInfo() {
    return {
      name: this.name,
      isReady: this.isReady,
      pointCount: 478,
      isRealModel: true,
      statusMessage: this.isReady ? `Operational (Face Landmarker, ${this.delegate})` : (this.loadError || "Loading"),
      architecture: "BlazeFace detector + face-mesh regression (Google)",
      processingType: "100% On-Device WebAssembly / WebGL"
    };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { MediaPipeDetector };
} else {
  window.MediaPipeDetector = MediaPipeDetector;
}
