/**
 * Faceify landmark labelling tool.
 *
 * An annotator places the 68 iBUG points on APAC faces prepared by `python -m src.apac_labels prepare`.
 * Every face starts from the mean WFLW face shape placed in a box around the face (never from a model
 * prediction). Labels are saved through src/serve_web.py (POST /api/labels) to data/apac_labels/ and
 * scored with `python -m src.apac_labels evaluate`. If the page is served without that endpoint, labels
 * are kept in this browser and can be downloaded.
 */
(function () {
  "use strict";

  const N = 68;
  const PRIORITY_GROUPS = ["East_Asian", "Southeast_Asian", "Indian", "White"];
  const CHAINS = [  // [first, last, closed]
    [0, 16, false], [17, 21, false], [22, 26, false], [27, 30, false], [31, 35, false],
    [36, 41, true], [42, 47, true], [48, 59, true], [60, 67, true]
  ];
  const REGIONS = [
    { name: "Jawline", from: 0, to: 16, color: "#f59e0b" },
    { name: "Left eyebrow (image left)", from: 17, to: 21, color: "#a855f7" },
    { name: "Right eyebrow (image right)", from: 22, to: 26, color: "#a855f7" },
    { name: "Nose", from: 27, to: 35, color: "#22c55e" },
    { name: "Left eye (image left)", from: 36, to: 41, color: "#06b6d4" },
    { name: "Right eye (image right)", from: 42, to: 47, color: "#06b6d4" },
    { name: "Outer lips", from: 48, to: 59, color: "#ef4444" },
    { name: "Inner lips", from: 60, to: 67, color: "#ec4899" }
  ];
  const LOCAL_KEY = "faceify-labels-v1";
  const NAME_KEY = "faceify-annotator";

  function describe(i) {
    if (i <= 16) {
      if (i === 0) return "Edge of the face on the image's left, at about eye level next to the ear (start of the jaw line).";
      if (i === 8) return "Chin tip: the lowest point of the chin, in the middle.";
      if (i === 16) return "Edge of the face on the image's right, at about eye level next to the ear (end of the jaw line).";
      return i < 8
        ? `Jaw line, image left side: point ${i + 1} of 17, evenly spaced between the ear and the chin.`
        : `Jaw line, image right side: point ${i + 1} of 17, evenly spaced between the chin and the ear.`;
    }
    const brow = { 17: "outer end", 18: "", 19: "highest middle", 20: "", 21: "inner end (towards the nose)",
                   22: "inner end (towards the nose)", 23: "", 24: "highest middle", 25: "", 26: "outer end" };
    if (i <= 26) {
      const side = i <= 21 ? "image-left" : "image-right";
      return `Upper edge of the ${side} eyebrow${brow[i] ? `: ${brow[i]}` : ", between its neighbours"}.`;
    }
    const nose = {
      27: "Top of the nose bridge, between the eyes (level with the eye corners' upper part).",
      28: "Nose bridge, one third of the way down.", 29: "Nose bridge, two thirds of the way down.",
      30: "Tip of the nose.", 31: "Bottom of the nose, outer edge of the image-left nostril.",
      32: "Bottom of the image-left nostril, half way to the middle.", 33: "Bottom of the nose, in the middle under the tip.",
      34: "Bottom of the image-right nostril, half way to the middle.", 35: "Bottom of the nose, outer edge of the image-right nostril."
    };
    if (nose[i]) return nose[i];
    const eye = (side, k) => [
      `Outer corner of the ${side} eye.`, `Upper eyelid of the ${side} eye, one third from the outer corner.`,
      `Upper eyelid of the ${side} eye, one third from the inner corner.`, `Inner corner of the ${side} eye (towards the nose).`,
      `Lower eyelid of the ${side} eye, one third from the inner corner.`, `Lower eyelid of the ${side} eye, one third from the outer corner.`][k];
    if (i <= 41) return eye("image-left", i - 36);
    if (i <= 47) {  // 42 inner, 43-44 upper, 45 outer, 46-47 lower
      return [`Inner corner of the image-right eye (towards the nose).`,
              `Upper eyelid of the image-right eye, one third from the inner corner.`,
              `Upper eyelid of the image-right eye, one third from the outer corner.`,
              `Outer corner of the image-right eye.`,
              `Lower eyelid of the image-right eye, one third from the outer corner.`,
              `Lower eyelid of the image-right eye, one third from the inner corner.`][i - 42];
    }
    const outer = {
      48: "Outer mouth corner, image left.", 49: "Upper lip outer edge, between the corner and the peak.",
      50: "Upper lip outer edge: image-left peak of the Cupid's bow.", 51: "Upper lip outer edge, in the middle (dip of the Cupid's bow).",
      52: "Upper lip outer edge: image-right peak of the Cupid's bow.", 53: "Upper lip outer edge, between the peak and the corner.",
      54: "Outer mouth corner, image right.", 55: "Lower lip outer edge, image right, near the corner.",
      56: "Lower lip outer edge, image right of the middle.", 57: "Lower lip outer edge, in the middle (lowest point).",
      58: "Lower lip outer edge, image left of the middle.", 59: "Lower lip outer edge, image left, near the corner."
    };
    if (outer[i]) return outer[i];
    return {
      60: "Inner mouth corner, image left (where the lips meet).", 61: "Inner edge of the upper lip, image left of the middle.",
      62: "Inner edge of the upper lip, in the middle.", 63: "Inner edge of the upper lip, image right of the middle.",
      64: "Inner mouth corner, image right.", 65: "Inner edge of the lower lip, image right of the middle.",
      66: "Inner edge of the lower lip, in the middle.", 67: "Inner edge of the lower lip, image left of the middle."
    }[i];
  }
  const regionOf = (i) => REGIONS.find((r) => i >= r.from && i <= r.to);

  // ------------------------------------------------------------------ state
  const S = {
    manifest: null, server: false, annotator: "", queue: [], pos: 0, item: null, img: null,
    points: [], init: [], occluded: [], current: 0, undo: [], mine: {}, labelledBy: {},
    view: { s: 1, ox: 0, oy: 0 }, drag: null, panMode: false, space: false, hover: null,
    started: 0, elapsed: 0, dirty: false
  };
  const $ = (id) => document.getElementById(id);
  const canvas = $("lbl-canvas"), ctx = canvas.getContext("2d");
  const loupe = $("lbl-loupe"), lctx = loupe.getContext("2d");

  function readLocal() { try { return JSON.parse(localStorage.getItem(LOCAL_KEY) || "{}"); } catch (e) { return {}; } }
  function writeLocal(v) { try { localStorage.setItem(LOCAL_KEY, JSON.stringify(v)); } catch (e) { /* private mode */ } }
  function status(msg, kind = "") { const el = $("lbl-status"); el.textContent = msg; el.dataset.kind = kind; }

  // ------------------------------------------------------------------ data
  async function loadManifest() {
    const r = await fetch("./labeling/manifest.json", { cache: "no-store" });
    if (!r.ok) throw new Error("labeling/manifest.json not found. Run: python -m src.apac_labels prepare");
    S.manifest = await r.json();
  }

  async function loadSaved() {
    S.mine = {}; S.labelledBy = {};
    try {
      const r = await fetch(`./api/labels?annotator=${encodeURIComponent(S.annotator)}`, { cache: "no-store" });
      if (!r.ok) throw new Error(String(r.status));
      const d = await r.json();
      S.server = true; S.mine = d.labels; S.labelledBy = d.labelled_by;
    } catch (e) {
      S.server = false;
      const local = readLocal()[S.annotator] || {};
      S.mine = local;
      for (const id of Object.keys(local)) if (!local[id].skipped) S.labelledBy[id] = [S.annotator];
    }
    const note = $("lbl-server-note");
    note.hidden = S.server;
    note.textContent = S.server ? "" :
      "Saving to the project needs the labelling server: python -m src.serve_web. Until then your labels are kept in this browser only - use \"Download my labels\".";
  }

  function buildQueue(mode) {
    const items = S.manifest.items;
    let pool;
    if (mode === "mine") pool = items.filter((it) => S.mine[it.id]);
    else if (mode === "second") pool = items.filter((it) => (S.labelledBy[it.id] || []).some((a) => a !== S.annotator) && !S.mine[it.id]);
    else {
      const groups = mode === "priority" ? PRIORITY_GROUPS : [...new Set(items.map((it) => it.group))];
      pool = items.filter((it) => groups.includes(it.group) && !S.mine[it.id] && !(S.labelledBy[it.id] || []).length);
    }
    // interleave groups so every group fills up at the same pace
    const byGroup = {};
    for (const it of pool) (byGroup[it.group] = byGroup[it.group] || []).push(it);
    const lists = Object.values(byGroup), out = [];
    for (let k = 0; lists.some((l) => k < l.length); k++) for (const l of lists) if (k < l.length) out.push(l[k]);
    return out;
  }

  function renderProgress() {
    const groups = {};
    for (const it of S.manifest.items) {
      const g = groups[it.group] = groups[it.group] || { label: it.group_label, total: 0, done: 0, mine: 0 };
      g.total++;
      if ((S.labelledBy[it.id] || []).length) g.done++;
      if (S.mine[it.id] && !S.mine[it.id].skipped) g.mine++;
    }
    const order = Object.keys(groups).sort((a, b) => (PRIORITY_GROUPS.indexOf(a) + 1 || 99) - (PRIORITY_GROUPS.indexOf(b) + 1 || 99));
    $("lbl-progress").innerHTML = order.map((k) => {
      const g = groups[k], pct = Math.round((100 * g.done) / g.total);
      return `<div class="lbl-prog ${PRIORITY_GROUPS.includes(k) ? "lbl-prog-priority" : ""}">
        <div class="lbl-prog-head"><span>${g.label}</span><span>${g.done} / ${g.total}</span></div>
        <div class="lbl-prog-track"><div class="lbl-prog-fill" style="width:${pct}%"></div></div>
        <div class="lbl-prog-sub">${g.mine} by you</div></div>`;
    }).join("");
  }

  // ------------------------------------------------------------------ face loading
  function loadFace(pos) {
    if (!S.queue.length) { status("Nothing left in this queue. Choose another queue above.", "ok"); return; }
    S.pos = Math.max(0, Math.min(pos, S.queue.length - 1));
    const it = S.item = S.queue[S.pos];
    const saved = S.mine[it.id];
    S.init = it.init.map((p) => [p[0], p[1]]);
    S.points = saved && saved.points ? saved.points.map((p) => [p[0], p[1]]) : S.init.map((p) => [p[0], p[1]]);
    S.occluded = saved && saved.occluded ? saved.occluded.slice() : new Array(N).fill(false);
    S.undo = []; S.current = 0; S.dirty = false;
    S.elapsed = saved ? saved.seconds || 0 : 0; S.started = performance.now();
    $("lbl-face-group").textContent = it.group_label;
    $("lbl-face-count").textContent = `Face ${S.pos + 1} of ${S.queue.length} · ${it.id}${saved ? (saved.skipped ? " · skipped before" : " · saved before") : ""}`;
    const img = new Image();
    img.onload = () => { S.img = img; fit(); updatePanel(); };
    img.onerror = () => status(`Image ${it.file} not found - re-run python -m src.apac_labels prepare`, "error");
    img.src = `./labeling/${it.file}`;
    status(saved ? "Loaded your saved points for this face." : "");
  }

  // ------------------------------------------------------------------ view
  function resizeCanvas() {
    const stage = $("lbl-stage"), dpr = window.devicePixelRatio || 1;
    const w = stage.clientWidth, h = stage.clientHeight;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`; canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function fit() {
    if (!S.img) return;
    resizeCanvas();
    const w = canvas.clientWidth, h = canvas.clientHeight;
    const s = Math.min(w / S.img.width, h / S.img.height) * 0.96;
    S.view = { s, ox: (w - S.img.width * s) / 2, oy: (h - S.img.height * s) / 2 };
    draw();
  }
  function zoomAt(factor, sx, sy) {
    const v = S.view, s = Math.max(0.2, Math.min(v.s * factor, 24));
    const ix = (sx - v.ox) / v.s, iy = (sy - v.oy) / v.s;
    S.view = { s, ox: sx - ix * s, oy: sy - iy * s };
    draw();
  }
  const toScreen = (p) => [p[0] * S.view.s + S.view.ox, p[1] * S.view.s + S.view.oy];
  const toImage = (x, y) => [(x - S.view.ox) / S.view.s, (y - S.view.oy) / S.view.s];

  function drawShape(c, map, radius, withNumbers) {
    if ($("lbl-lines").checked) {
      c.lineWidth = 1.2;
      for (const [a, b, closed] of CHAINS) {
        c.strokeStyle = regionOf(a).color + "aa";
        c.beginPath();
        for (let i = a; i <= b; i++) { const [x, y] = map(S.points[i]); i === a ? c.moveTo(x, y) : c.lineTo(x, y); }
        if (closed) c.closePath();
        c.stroke();
      }
    }
    for (let i = 0; i < N; i++) {
      const [x, y] = map(S.points[i]);
      const isCur = i === S.current;
      c.beginPath();
      c.arc(x, y, isCur ? radius * 1.6 : radius, 0, 2 * Math.PI);
      c.fillStyle = regionOf(i).color;
      c.strokeStyle = "#ffffff";
      c.lineWidth = isCur ? 2 : 1;
      if (S.occluded[i]) { c.stroke(); c.strokeStyle = regionOf(i).color; c.lineWidth = 1.5; c.stroke(); }
      else { c.fill(); c.stroke(); }
      if (isCur) {
        c.beginPath(); c.arc(x, y, radius * 3.2, 0, 2 * Math.PI);
        c.strokeStyle = "#ffffff"; c.lineWidth = 1.5; c.setLineDash([3, 3]); c.stroke(); c.setLineDash([]);
      }
      if (withNumbers) {
        c.font = "600 10px 'DM Sans', sans-serif"; c.fillStyle = "#ffffff";
        c.strokeStyle = "rgba(10,15,26,0.8)"; c.lineWidth = 3;
        c.strokeText(String(i + 1), x + radius + 2, y - radius - 1); c.fillText(String(i + 1), x + radius + 2, y - radius - 1);
      }
    }
  }

  function draw() {
    if (!S.img) return;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    ctx.clearRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = S.view.s < 3;
    ctx.drawImage(S.img, S.view.ox, S.view.oy, S.img.width * S.view.s, S.img.height * S.view.s);
    drawShape(ctx, toScreen, 3.5, $("lbl-numbers").checked);
    drawLoupe();
  }

  function drawLoupe() {
    const at = S.drag && S.drag.point != null ? S.points[S.drag.point] : S.hover;
    if (!at || !S.img) { loupe.style.display = "none"; return; }
    loupe.style.display = "block";
    const L = loupe.width, zoom = 4, half = L / (2 * zoom);
    lctx.save();
    lctx.fillStyle = "#0a0f1a"; lctx.fillRect(0, 0, L, L);
    lctx.imageSmoothingEnabled = false;
    lctx.drawImage(S.img, at[0] - half, at[1] - half, 2 * half, 2 * half, 0, 0, L, L);
    drawShape(lctx, (p) => [(p[0] - at[0] + half) * zoom, (p[1] - at[1] + half) * zoom], 3, false);
    lctx.strokeStyle = "rgba(255,255,255,0.8)"; lctx.lineWidth = 1;
    lctx.beginPath(); lctx.moveTo(L / 2, 0); lctx.lineTo(L / 2, L); lctx.moveTo(0, L / 2); lctx.lineTo(L, L / 2); lctx.stroke();
    lctx.restore();
  }

  // ------------------------------------------------------------------ editing
  function pushUndo() { S.undo.push({ points: S.points.map((p) => p.slice()), occluded: S.occluded.slice(), current: S.current }); if (S.undo.length > 200) S.undo.shift(); }
  function movedCount() { return S.points.filter((p, i) => Math.hypot(p[0] - S.init[i][0], p[1] - S.init[i][1]) > 0.5).length; }
  function select(i) { S.current = (i + N) % N; updatePanel(); draw(); }

  function updatePanel() {
    const i = S.current, r = regionOf(i);
    $("lbl-point-num").textContent = i + 1;
    $("lbl-point-region").textContent = r.name;
    $("lbl-point-region").style.color = r.color;
    $("lbl-point-desc").textContent = describe(i);
    $("lbl-occluded").checked = S.occluded[i];
    $("lbl-moved").textContent = `${movedCount()} / ${N}`;
    drawReference();
  }

  function drawReference() {
    const m = S.manifest.mean_shape;
    let svg = "";
    for (const [a, b, closed] of CHAINS) {
      const pts = [];
      for (let k = a; k <= b; k++) pts.push(`${m[k][0].toFixed(3)},${m[k][1].toFixed(3)}`);
      svg += `<polyline points="${pts.join(" ")}${closed ? " " + pts[0] : ""}" fill="none" stroke="#c9c2b4" stroke-width="0.006"/>`;
    }
    for (let k = 0; k < N; k++) {
      const cur = k === S.current;
      svg += `<circle cx="${m[k][0]}" cy="${m[k][1]}" r="${cur ? 0.03 : 0.011}" fill="${cur ? regionOf(k).color : "#8f877a"}" ${cur ? 'stroke="#0a0f1a" stroke-width="0.006"' : ""}/>`;
    }
    $("lbl-ref").innerHTML = svg;
  }

  function tick() {
    if (S.item && document.visibilityState === "visible") {
      const secs = S.elapsed + (performance.now() - S.started) / 1000;
      $("lbl-timer").textContent = `${Math.floor(secs / 60)}:${String(Math.floor(secs % 60)).padStart(2, "0")}`;
    }
  }
  const seconds = () => Math.round(S.elapsed + (performance.now() - S.started) / 1000);

  // ------------------------------------------------------------------ pointer input
  function hit(sx, sy) {
    let best = null, bestD = 9;
    for (let i = 0; i < N; i++) {
      const [x, y] = toScreen(S.points[i]);
      const d = Math.hypot(x - sx, y - sy);
      if (d < bestD) { best = i; bestD = d; }
    }
    return best;
  }
  function local(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener("pointerdown", (e) => {
    if (!S.img) return;
    canvas.setPointerCapture(e.pointerId);
    const [sx, sy] = local(e);
    const pan = S.panMode || S.space || e.button === 1 || e.button === 2;
    const p = pan ? null : hit(sx, sy);
    S.drag = { sx, sy, ox: S.view.ox, oy: S.view.oy, pan, point: p, moved: false };
    if (p != null) { pushUndo(); select(p); }
  });
  canvas.addEventListener("pointermove", (e) => {
    const [sx, sy] = local(e);
    S.hover = S.img ? toImage(sx, sy) : null;
    const d = S.drag;
    if (d) {
      if (Math.hypot(sx - d.sx, sy - d.sy) > 3) d.moved = true;
      if (d.pan) { S.view.ox = d.ox + sx - d.sx; S.view.oy = d.oy + sy - d.sy; }
      else if (d.point != null) { S.points[d.point] = toImage(sx, sy); S.dirty = true; $("lbl-moved").textContent = `${movedCount()} / ${N}`; }
    }
    draw();
  });
  canvas.addEventListener("pointerup", (e) => {
    const d = S.drag; S.drag = null;
    if (!d) return;
    if (!d.pan && d.point == null && !d.moved) {   // click on empty space: place the current point, advance
      pushUndo();
      const [sx, sy] = local(e);
      S.points[S.current] = toImage(sx, sy);
      S.dirty = true;
      select(S.current + 1);
    }
    updatePanel(); draw();
  });
  canvas.addEventListener("pointerleave", () => { S.hover = null; draw(); });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    const [sx, sy] = local(e);
    zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, sx, sy);
  }, { passive: false });

  // ------------------------------------------------------------------ saving
  async function save(skipped) {
    const it = S.item;
    if (!it) return false;
    if (!skipped && movedCount() < 40 &&
        !confirm(`Only ${movedCount()} of 68 points were moved from the starting shape. Every point should be checked. Save anyway?`)) return false;
    const label = { id: it.id, annotator: S.annotator, skipped, seconds: seconds() };
    if (skipped) label.skip_reason = $("lbl-skip-reason").value;
    else Object.assign(label, { points: S.points.map((p) => [+p[0].toFixed(2), +p[1].toFixed(2)]), occluded: S.occluded, moved: movedCount() });
    if (S.server) {
      try {
        const r = await fetch("./api/labels", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(label) });
        const d = await r.json();
        if (!r.ok) throw new Error(d.error || r.status);
        S.mine[it.id] = d.record;
        status(`Saved to ${d.saved}`, "ok");
      } catch (e) {
        status(`Not saved: ${e.message}`, "error");
        return false;
      }
    } else {
      const all = readLocal();
      all[S.annotator] = all[S.annotator] || {};
      all[S.annotator][it.id] = { ...label, saved: new Date().toISOString() };
      writeLocal(all);
      S.mine[it.id] = all[S.annotator][it.id];
      status("Saved in this browser (start python -m src.serve_web to save to the project).", "warn");
    }
    if (!skipped) S.labelledBy[it.id] = [...new Set([...(S.labelledBy[it.id] || []), S.annotator])];
    S.dirty = false;
    renderProgress();
    return true;
  }
  async function saveAndNext(skipped) {
    if (await save(skipped)) {
      const msg = $("lbl-status").textContent, kind = $("lbl-status").dataset.kind;
      if (S.pos + 1 < S.queue.length) { loadFace(S.pos + 1); status(`Previous face: ${msg}`, kind); }
      else status("That was the last face in this queue. Thank you!", "ok");
    }
  }
  function leaveFace(delta) {
    if (S.dirty && !confirm("This face has unsaved changes. Leave it without saving?")) return;
    loadFace(S.pos + delta);
  }

  async function exportLabels() {
    let data = S.mine;
    if (S.server) {
      try { data = (await (await fetch(`./api/labels?annotator=${encodeURIComponent(S.annotator)}`)).json()).labels; } catch (e) { /* use cached */ }
    }
    const blob = new Blob([JSON.stringify({ annotator: S.annotator, layout: S.manifest.layout, labels: data }, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `apac_labels_${S.annotator}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  // ------------------------------------------------------------------ controls
  function bind() {
    $("lbl-zoom-in").onclick = () => zoomAt(1.3, canvas.clientWidth / 2, canvas.clientHeight / 2);
    $("lbl-zoom-out").onclick = () => zoomAt(1 / 1.3, canvas.clientWidth / 2, canvas.clientHeight / 2);
    $("lbl-zoom-fit").onclick = fit;
    $("lbl-pan").onclick = () => { S.panMode = !S.panMode; $("lbl-pan").classList.toggle("active", S.panMode); };
    $("lbl-lines").onchange = draw;
    $("lbl-numbers").onchange = draw;
    $("lbl-prev-pt").onclick = () => select(S.current - 1);
    $("lbl-next-pt").onclick = () => select(S.current + 1);
    $("lbl-occluded").onchange = (e) => { pushUndo(); S.occluded[S.current] = e.target.checked; S.dirty = true; draw(); };
    $("lbl-undo").onclick = undo;
    $("lbl-reset").onclick = () => { pushUndo(); S.points = S.init.map((p) => p.slice()); S.occluded.fill(false); S.dirty = true; select(0); };
    $("lbl-save").onclick = () => saveAndNext(false);
    $("lbl-skip").onclick = () => saveAndNext(true);
    $("lbl-prev-face").onclick = () => leaveFace(-1);
    $("lbl-next-face").onclick = () => leaveFace(1);
    $("lbl-export").onclick = exportLabels;
    $("lbl-start").onclick = start;
    window.addEventListener("resize", () => { if (S.img) fit(); });
    window.addEventListener("beforeunload", (e) => { if (S.dirty) { e.preventDefault(); e.returnValue = ""; } });
    document.addEventListener("keydown", onKey);
    document.addEventListener("keyup", (e) => { if (e.code === "Space") S.space = false; });
  }
  function undo() {
    const u = S.undo.pop();
    if (!u) return;
    S.points = u.points; S.occluded = u.occluded; S.current = u.current; S.dirty = true;
    updatePanel(); draw();
  }
  function onKey(e) {
    if (!S.item || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) && document.activeElement.type !== "checkbox") return;
    const step = e.shiftKey ? 2 : 0.5;
    const nudge = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[e.key];
    if (nudge) {
      e.preventDefault(); pushUndo();
      S.points[S.current] = [S.points[S.current][0] + nudge[0], S.points[S.current][1] + nudge[1]];
      S.dirty = true; updatePanel(); draw(); return;
    }
    if (e.code === "Space") { S.space = true; e.preventDefault(); return; }
    if (e.key === "Tab") { e.preventDefault(); select(S.current + (e.shiftKey ? -1 : 1)); return; }
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); undo(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const k = e.key.toLowerCase();
    if (k === "n") select(S.current + 1);
    else if (k === "p") select(S.current - 1);
    else if (k === "o") { pushUndo(); S.occluded[S.current] = !S.occluded[S.current]; S.dirty = true; updatePanel(); draw(); }
    else if (k === "f") fit();
    else if (k === "+" || k === "=") zoomAt(1.3, canvas.clientWidth / 2, canvas.clientHeight / 2);
    else if (k === "-") zoomAt(1 / 1.3, canvas.clientWidth / 2, canvas.clientHeight / 2);
    else if (e.key === "Enter") { e.preventDefault(); saveAndNext(false); }
  }

  async function start() {
    const name = $("lbl-annotator").value.trim();
    if (!/^[A-Za-z0-9_-]{1,40}$/.test(name)) { alert("Enter your name using letters, digits, - or _ (no spaces)."); return; }
    if (S.dirty && !confirm("This face has unsaved changes. Discard them?")) return;
    S.annotator = name;
    try { localStorage.setItem(NAME_KEY, name); } catch (e) { /* private mode */ }
    await loadSaved();
    S.queue = buildQueue($("lbl-queue").value);
    renderProgress();
    $("lbl-workspace").hidden = false;
    loadFace(0);
    if (!S.queue.length) status("Nothing left in this queue. Choose another queue above.", "ok");
    $("lbl-workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  async function init() {
    bind();
    try { $("lbl-annotator").value = localStorage.getItem(NAME_KEY) || ""; } catch (e) { /* private mode */ }
    try {
      await loadManifest();
      S.annotator = $("lbl-annotator").value.trim() || "_";
      await loadSaved();
      renderProgress();
    } catch (e) {
      $("lbl-setup").insertAdjacentHTML("beforeend", `<div class="lbl-server-note">${e.message}</div>`);
      $("lbl-start").disabled = true;
    }
    setInterval(tick, 500);
  }
  init();
})();
