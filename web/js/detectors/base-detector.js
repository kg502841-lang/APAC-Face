/**
 * Faceify AI Facial Landmark Analysis - Base Detector Interface
 * -------------------------------------------------------------
 * Abstract base class defining the contract for all facial landmark detectors.
 * Follows clean object-oriented architecture to allow seamless swapping between
 * the baseline detector (MediaPipe) and custom trained models (APAC Detector).
 */

class LandmarkDetector {
  constructor(name, options = {}) {
    if (new.target === LandmarkDetector) {
      throw new TypeError("Cannot construct LandmarkDetector instances directly.");
    }
    this.name = name;
    this.options = options;
    this.isReady = false;
    this.pointCount = 0;
  }

  /**
   * Initializes model weights, wasm runtimes, or backend endpoints.
   * @returns {Promise<boolean>}
   */
  async initialize() {
    throw new Error("Method 'initialize()' must be implemented.");
  }

  /**
   * Detects facial landmarks on an HTMLImageElement, HTMLVideoElement, or HTMLCanvasElement.
   * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} inputElement
   * @returns {Promise<LandmarkResult>}
   */
  async detect(inputElement) {
    throw new Error("Method 'detect()' must be implemented.");
  }

  /**
   * Returns total landmark count produced by this detector (e.g., 468, 68).
   * @returns {number}
   */
  getPointCount() {
    return this.pointCount;
  }

  /**
   * Returns metadata and deployment status of the model.
   * @returns {Object}
   */
  getModelInfo() {
    return {
      name: this.name,
      isReady: this.isReady,
      pointCount: this.pointCount,
      isRealModel: true,
      statusMessage: "Operational"
    };
  }
}

// Export for module or global window scope
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { LandmarkDetector };
} else {
  window.LandmarkDetector = LandmarkDetector;
}
