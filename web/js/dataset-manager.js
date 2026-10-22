/**
 * Faceify AI Facial Landmark Analysis - Dataset & Metrics Manager
 * ---------------------------------------------------------------
 * Loads the benchmark galleries, parses annotation files (TXT/.pts, CSV, JSON) and computes error
 * metrics (Euclidean error, inter-ocular NME, maximum error, failure rate).
 *
 * Ground truth is ONLY ever real annotation data:
 *   - WFLW test faces exported by `python -m src.export_web` (manual 98-point annotations,
 *     compared on the common 68-point layout), or
 *   - an annotation file the user uploads for their own image.
 * Photos without annotations (webcam, uploads, the APAC gallery) get no error metrics.
 *
 * All models are compared on the same 68-point iBUG layout:
 *   MediaPipe 478-vertex mesh -> 68 via MEDIAPIPE_TO_68_MAP, APAC-Face 98 WFLW points -> 68 via WFLW_TO_68_MAP.
 */

class DatasetManager {
  constructor() {
    // MediaPipe face-mesh vertex indices mapped to the standard 68 iBUG landmarks
    this.MEDIAPIPE_TO_68_MAP = [
      162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,  // jaw
      70, 63, 105, 66, 107,          // right eyebrow
      336, 296, 334, 293, 300,       // left eyebrow
      168, 197, 5, 4,                // nose bridge
      75, 97, 2, 326, 305,           // lower nose
      33, 160, 158, 133, 153, 144,   // right eye
      362, 385, 387, 263, 373, 380,  // left eye
      61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,  // outer lips
      78, 81, 13, 311, 308, 402, 14, 178                    // inner lips
    ];
    // WFLW 98-point -> 68 iBUG (standard subset)
    this.WFLW_TO_68_MAP = [
      0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32,
      33, 34, 35, 36, 37, 42, 43, 44, 45, 46,
      51, 52, 53, 54, 55, 56, 57, 58, 59,
      60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75,
      76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
      88, 89, 90, 91, 92, 93, 94, 95
    ];
    this.POINT_FAILURE_THRESHOLD = 0.08;

    this.wflwSamples = [];
    this.apacSamples = [];
    this.allSamples = [];
    this.groundTruthCache = new Map();
    this.loadNotes = [];
  }

  /** Loads the WFLW (ground truth) and APAC (no ground truth) galleries. */
  async loadFullCatalog() {
    const load = async (path) => {
      try {
        const resp = await fetch(path);
        return resp.ok ? await resp.json() : null;
      } catch (e) {
        return null;
      }
    };
    const wflw = await load("./data/wflw_benchmark.json");
    const apac = await load("./data/apac_gallery.json");
    this.wflwSamples = wflw ? wflw.samples : [];
    this.apacSamples = apac ? apac.samples : [];
    if (!wflw) this.loadNotes.push("WFLW ground-truth gallery not exported yet (python -m src.export_web).");
    if (!apac) this.loadNotes.push("APAC photo gallery not exported yet (python -m src.export_web --apac-source ...).");
    for (const s of this.wflwSamples) {
      this.groundTruthCache.set(s.key, this._groundTruthFromSample(s));
    }
    this.allSamples = [...this.wflwSamples, ...this.apacSamples];
    return this.allSamples.length > 0;
  }

  _groundTruthFromSample(sample) {
    const landmarks = sample.gt68.map(([x, y], idx) => ({ id: idx + 1, x, y, region: this.getRegionName(idx) }));
    return {
      landmarks,
      pointCount: 68,
      iod: this.calculateInterOcularDistance(landmarks),
      source: sample.source || "WFLW test (manual annotation)"
    };
  }

  categoryCounts() {
    const isApacRegion = (r) => ["East Asia", "South Asia", "Southeast Asia"].includes(r);
    return {
      all: this.allSamples.length,
      gt: this.wflwSamples.length,
      apac: this.apacSamples.filter(s => s.is_apac).length,
      "East Asia": this.apacSamples.filter(s => s.region === "East Asia").length,
      "South Asia": this.apacSamples.filter(s => s.region === "South Asia").length,
      "Southeast Asia": this.apacSamples.filter(s => s.region === "Southeast Asia").length,
      other: this.apacSamples.filter(s => !isApacRegion(s.region)).length
    };
  }

  getFilteredSamples(filter = 'all', query = '', page = 1, pageSize = 12) {
    let list = this.allSamples;
    const isApacRegion = (r) => ["East Asia", "South Asia", "Southeast Asia"].includes(r);
    if (filter === 'gt') list = this.wflwSamples;
    else if (filter === 'apac') list = this.apacSamples.filter(s => s.is_apac);
    else if (filter === 'other') list = this.apacSamples.filter(s => !isApacRegion(s.region));
    else if (filter !== 'all') list = this.apacSamples.filter(s => s.region === filter);

    if (query && query.trim().length > 0) {
      const q = query.toLowerCase().trim();
      list = list.filter(s =>
        (s.id && s.id.toLowerCase().includes(q)) ||
        (s.name && s.name.toLowerCase().includes(q)) ||
        (s.region && s.region.toLowerCase().includes(q)) ||
        (s.conditions && s.conditions.join(" ").toLowerCase().includes(q)) ||
        (s.file && s.file.toLowerCase().includes(q)));
    }

    const totalFiltered = list.length;
    const totalPages = Math.max(1, Math.ceil(totalFiltered / pageSize));
    const safePage = Math.min(Math.max(1, page), totalPages);
    const startIndex = (safePage - 1) * pageSize;
    return {
      items: list.slice(startIndex, startIndex + pageSize),
      totalCount: totalFiltered,
      currentPage: safePage,
      totalPages,
      startIndex: totalFiltered ? startIndex + 1 : 0,
      endIndex: Math.min(startIndex + pageSize, totalFiltered)
    };
  }

  /** Anatomical region of a point in the 68-point layout. */
  getRegionName(zeroBasedIndex) {
    if (zeroBasedIndex < 17) return "Jawline";
    if (zeroBasedIndex < 22) return "Right Eyebrow";
    if (zeroBasedIndex < 27) return "Left Eyebrow";
    if (zeroBasedIndex < 36) return "Nose";
    if (zeroBasedIndex < 42) return "Right Eye";
    if (zeroBasedIndex < 48) return "Left Eye";
    return "Mouth";
  }

  /** Region of point `index` for a prediction with `pointCount` points. */
  getRegionForLayout(index, pointCount) {
    if (pointCount === 68) return this.getRegionName(index);
    if (pointCount === 98 && window.APACDetector) return APACDetector.regionName(index);
    return "Face Mesh";
  }

  /** Real ground truth for a gallery sample, or null when the sample has none. */
  async getGroundTruth(sampleKey) {
    return this.groundTruthCache.get(sampleKey) || null;
  }

  parsePtsTxt(txtContent) {
    const lines = txtContent.split(/\r?\n/);
    const points = [];
    let insideBlock = false;
    for (const line of lines) {
      const trimmed = line.trim();
      if (trimmed === "{") { insideBlock = true; continue; }
      if (trimmed === "}") break;
      if (insideBlock && trimmed.length > 0) {
        const parts = trimmed.split(/\s+/);
        const x = parseFloat(parts[0]);
        const y = parseFloat(parts[1]);
        if (!isNaN(x) && !isNaN(y)) {
          const idx = points.length;
          points.push({ id: idx + 1, x, y, region: this.getRegionName(idx) });
        }
      }
    }
    return { landmarks: points, pointCount: points.length, iod: this.calculateInterOcularDistance(points), source: "Uploaded annotation" };
  }

  parseCsv(csvContent) {
    const lines = csvContent.trim().split(/\r?\n/);
    if (lines.length <= 1) return null;
    const points = [];
    for (let i = 1; i < lines.length; i++) {
      const parts = lines[i].split(",");
      if (parts.length >= 4) {
        const x = parseFloat(parts[2]);
        const y = parseFloat(parts[3]);
        if (!isNaN(x) && !isNaN(y)) {
          points.push({ id: parseInt(parts[1], 10) || points.length + 1, x, y, region: this.getRegionName(points.length) });
        }
      }
    }
    return { landmarks: points, pointCount: points.length, iod: this.calculateInterOcularDistance(points), source: "Uploaded annotation" };
  }

  parseJson(content) {
    const data = typeof content === "string" ? JSON.parse(content) : content;
    const raw = data.landmarks || data.gt68 || data;
    const points = raw.map((p, idx) => Array.isArray(p)
      ? { id: idx + 1, x: +p[0], y: +p[1], region: this.getRegionName(idx) }
      : { id: p.id || idx + 1, x: +p.x, y: +p.y, region: this.getRegionName(idx) });
    return { landmarks: points, pointCount: points.length, iod: this.calculateInterOcularDistance(points), source: "Uploaded annotation" };
  }

  calculateInterOcularDistance(points) {
    if (!points || points.length < 46) return null;
    const a = points[36];
    const b = points[45];
    return Math.hypot(b.x - a.x, b.y - a.y);
  }

  /** Reduce any supported prediction layout to the 68-point layout. */
  mapTo68(predicted) {
    if (!predicted || predicted.length === 0) return [];
    let map = null;
    if (predicted.length >= 468) map = this.MEDIAPIPE_TO_68_MAP;
    else if (predicted.length === 98) map = this.WFLW_TO_68_MAP;
    if (!map) return predicted.slice(0, 68);
    return map.map((idx, i) => ({ id: i + 1, x: predicted[idx].x, y: predicted[idx].y }));
  }

  /**
   * Euclidean error, NME, max error and failure rate of a prediction against 68-point ground truth.
   * Returns null when there is no usable ground truth (never an estimate).
   */
  calculateMetrics(predictedLandmarks, groundTruthLandmarks) {
    if (!predictedLandmarks || !predictedLandmarks.length || !groundTruthLandmarks || groundTruthLandmarks.length !== 68) {
      return null;
    }
    const mapped = this.mapTo68(predictedLandmarks);
    const iod = this.calculateInterOcularDistance(groundTruthLandmarks);
    if (!iod || iod <= 0 || mapped.length !== 68) return null;

    let sum = 0, maxError = -Infinity, failures = 0;
    const pointErrors = [];
    for (let i = 0; i < 68; i++) {
      const pred = mapped[i];
      const gt = groundTruthLandmarks[i];
      const d = Math.hypot(pred.x - gt.x, pred.y - gt.y);
      sum += d;
      if (d > maxError) maxError = d;
      if (d / iod > this.POINT_FAILURE_THRESHOLD) failures++;
      pointErrors.push({
        id: i + 1, region: this.getRegionName(i), euclideanPx: +d.toFixed(2),
        normalizedErrorPercent: +(d / iod * 100).toFixed(2),
        predX: pred.x, predY: pred.y, gtX: gt.x, gtY: gt.y
      });
    }
    const mean = sum / 68;
    return {
      pointsCompared: 68,
      meanErrorPx: +mean.toFixed(2),
      maxErrorPx: +maxError.toFixed(2),
      nmePercent: +(mean / iod * 100).toFixed(2),
      failureRatePercent: +(failures / 68 * 100).toFixed(1),
      iodPx: +iod.toFixed(1),
      normalizationMethod: "Inter-Ocular Distance (Outer Eye Corners)",
      pointErrors,
      mappedPredictions: mapped
    };
  }

  /** Mean distance between two models' 68-point predictions / their inter-ocular distance (not accuracy). */
  calculateAgreement(predA, predB) {
    const a = this.mapTo68(predA);
    const b = this.mapTo68(predB);
    if (a.length !== 68 || b.length !== 68) return null;
    const mid = a.map((p, i) => ({ x: (p.x + b[i].x) / 2, y: (p.y + b[i].y) / 2 }));
    const iod = this.calculateInterOcularDistance(mid);
    let sum = 0;
    for (let i = 0; i < 68; i++) sum += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
    return { meanPx: +(sum / 68).toFixed(2), percentOfIod: +(sum / 68 / iod * 100).toFixed(2) };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = { DatasetManager };
} else {
  window.DatasetManager = DatasetManager;
}
