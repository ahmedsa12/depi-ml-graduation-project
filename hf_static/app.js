/* Real-Time Sign Language Recognition — 100% in-browser (Hugging Face Static Space).
   MediaPipe HandLandmarker (WASM) + the static MLP re-implemented in JS.
   No server: the webcam never leaves the device. */

import { FilesetResolver, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17],
];

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), overlay: $("overlay"), hint: $("videoHint"),
  letter: $("letter"), source: $("sourceTag"),
  conf: $("confBar"), confVal: $("confVal"), stab: $("stabBar"), stabVal: $("stabVal"),
  fps: $("fps"), hand: $("hand"), word: $("word"),
  statusDot: $("statusDot"), statusText: $("statusText"),
  start: $("startBtn"), stop: $("stopBtn"),
  space: $("spaceBtn"), back: $("backBtn"), clear: $("clearBtn"),
};
const octx = els.overlay.getContext("2d");

function setStatus(state, text) {
  els.statusDot.className = "dot " + ({ on: "dot-on", off: "dot-off", err: "dot-err" }[state]);
  els.statusText.textContent = text;
}

// ------------------------------------------------------------------ MLP (JS)
let M = null;   // weights payload
function relu(a) { for (let i = 0; i < a.length; i++) if (a[i] < 0) a[i] = 0; return a; }
function dense(x, layer) {            // x: Float64Array(in) -> Float64Array(out)
  const { w, b, in: nin, out: nout } = layer;
  const y = new Float64Array(nout);
  for (let j = 0; j < nout; j++) y[j] = b[j];
  for (let i = 0; i < nin; i++) {
    const xi = x[i]; if (xi === 0) continue;
    const base = i * nout;
    for (let j = 0; j < nout; j++) y[j] += xi * w[base + j];
  }
  return y;
}
function bn(x, p, eps) {
  const { gamma, beta, mean, var: v } = p;
  for (let i = 0; i < x.length; i++) x[i] = gamma[i] * (x[i] - mean[i]) / Math.sqrt(v[i] + eps) + beta[i];
  return x;
}
function softmax(x) {
  let m = -Infinity; for (const v of x) if (v > m) m = v;
  let s = 0; const o = new Float64Array(x.length);
  for (let i = 0; i < x.length; i++) { o[i] = Math.exp(x[i] - m); s += o[i]; }
  for (let i = 0; i < x.length; i++) o[i] /= s;
  return o;
}
function scale(feat) {                // StandardScaler
  const o = new Float64Array(feat.length);
  for (let i = 0; i < feat.length; i++) o[i] = (feat[i] - M.scaler_mean[i]) / M.scaler_scale[i];
  return o;
}
function predictProbs(feat63) {
  let x = scale(feat63);
  x = relu(bn(dense(x, M.dense[0]), M.bn[0], M.bn_eps));
  x = relu(bn(dense(x, M.dense[1]), M.bn[1], M.bn_eps));
  x = relu(bn(dense(x, M.dense[2]), M.bn[2], M.bn_eps));
  return softmax(dense(x, M.dense[3]));
}
function predict(feat63) {
  const probs = predictProbs(feat63);
  let bi = 0; for (let i = 1; i < probs.length; i++) if (probs[i] > probs[bi]) bi = i;
  return { label: M.classes[bi], conf: probs[bi] };
}

// wrist-relative + scale-invariant normalization -> Float64Array(63)
function normalize(landmarks) {
  const w0 = landmarks[0];
  const c = landmarks.map((p) => [p.x - w0.x, p.y - w0.y, p.z - w0.z]);
  let maxn = 0;
  for (const p of c) { const n = Math.hypot(p[0], p[1], p[2]); if (n > maxn) maxn = n; }
  const s = maxn > 1e-6 ? maxn : 1;
  const out = new Float64Array(63);
  for (let i = 0; i < 21; i++) { out[i*3] = c[i][0]/s; out[i*3+1] = c[i][1]/s; out[i*3+2] = c[i][2]/s; }
  return out;
}

// ------------------------------------------------------------------ word builder
class WordBuilder {
  constructor(stab) { this.stab = stab; this.reset(); }
  reset() { this.text = ""; this.cand = null; this.count = 0; this.locked = false; }
  handMissing() { this.cand = null; this.count = 0; this.locked = false; }
  update(tok) {
    if (tok !== this.cand) { this.cand = tok; this.count = 1; } else { this.count++; }
    if (this.locked) return this.text;
    if (this.count < this.stab) return this.text;
    if (tok === "space") this.text += " ";
    else if (tok === "del" || tok === "delete") this.text = this.text.slice(0, -1);
    else this.text += tok;
    this.locked = true;
    return this.text;
  }
}

// ------------------------------------------------------------------ state
let landmarker = null, stream = null, running = false;
let smooth = [], word = null, lastVideoTime = -1;
let fpsEma = 0, lastFrame = 0;

function majority(arr) {
  const m = {}; let best = arr[0], bestc = 0;
  for (const x of arr) { m[x] = (m[x] || 0) + 1; if (m[x] > bestc) { bestc = m[x]; best = x; } }
  return { label: best, frac: bestc / arr.length };
}

// ------------------------------------------------------------------ init
async function init() {
  const res = await fetch("weights.json");
  M = await res.json();
  M.dense.forEach((d) => { d.w = Float64Array.from(d.w); d.b = Float64Array.from(d.b); });
  M.bn.forEach((p) => { for (const k of ["gamma","beta","mean","var"]) p[k] = Float64Array.from(p[k]); });
  M.scaler_mean = Float64Array.from(M.scaler_mean);
  M.scaler_scale = Float64Array.from(M.scaler_scale);

  // self-test: JS forward must match the exported numpy forward
  const out = softmaxTest(M.test_input);
  let maxDiff = 0;
  for (let i = 0; i < out.length; i++) maxDiff = Math.max(maxDiff, Math.abs(out[i] - M.test_output[i]));
  console.log("MLP self-test max |JS - numpy| =", maxDiff.toExponential(2),
              maxDiff < 1e-4 ? "PASS ✅" : "FAIL ❌");

  word = new WordBuilder(M.stability_frames);
  setStatus("off", "Loading hand model…");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  try {
    landmarker = await createLM(vision, "GPU");
  } catch (e) {
    console.warn("GPU delegate failed, using CPU", e);
    landmarker = await createLM(vision, "CPU");
  }
  setStatus("off", "Ready");
  els.hint.textContent = "Click Start camera to begin";
  els.start.disabled = false;
}

function createLM(vision, delegate) {
  return HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: "hand_landmarker.task", delegate },
    numHands: 1, runningMode: "VIDEO",
    minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
}

function softmaxTest(feat) {  // run the JS pipeline on the raw test vector (already 63-D)
  return predictProbs(Float64Array.from(feat));
}

// ------------------------------------------------------------------ camera loop
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    els.video.srcObject = stream;
    await els.video.play();
    els.hint.style.display = "none";
    sizeOverlay();
    running = true; smooth = []; word.reset(); els.word.textContent = "";
    els.start.disabled = true; els.stop.disabled = false;
    [els.space, els.back, els.clear].forEach((b) => (b.disabled = false));
    setStatus("on", "Live");
    requestAnimationFrame(loop);
  } catch (e) {
    setStatus("err", "Camera blocked");
    alert("Could not access the webcam: " + e.message);
  }
}

function stopCamera() {
  running = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  els.start.disabled = false; els.stop.disabled = true;
  [els.space, els.back, els.clear].forEach((b) => (b.disabled = true));
  octx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.hint.style.display = "grid"; setStatus("off", "Stopped");
  els.letter.textContent = "–"; els.source.textContent = "idle"; els.source.className = "source-tag";
}

function sizeOverlay() { els.overlay.width = els.video.clientWidth; els.overlay.height = els.video.clientHeight; }
window.addEventListener("resize", () => running && sizeOverlay());

function loop() {
  if (!running) return;
  const now = performance.now();
  if (els.video.currentTime !== lastVideoTime) {
    lastVideoTime = els.video.currentTime;
    const result = landmarker.detectForVideo(els.video, now);
    handleResult(result);
  }
  if (lastFrame) { const inst = 1000 / (now - lastFrame); fpsEma = fpsEma ? fpsEma*0.8 + inst*0.2 : inst; els.fps.textContent = fpsEma.toFixed(0); }
  lastFrame = now;
  requestAnimationFrame(loop);
}

function handleResult(result) {
  const has = result.landmarks && result.landmarks.length > 0;
  const N = M.smoothing_window;
  if (!has) {
    smooth.push("nothing"); if (smooth.length > N) smooth.shift();
    word.handMissing();
    draw(null); updateUI("–", 0, majority(smooth).frac, false, word.text);
    return;
  }
  const lm = result.landmarks[0];
  const { label, conf } = predict(normalize(lm));
  let token = conf < M.min_confidence ? "nothing" : label;
  smooth.push(token); if (smooth.length > N) smooth.shift();
  const sm = majority(smooth);
  if (sm.label === "nothing") { word.handMissing(); }
  else { word.update(sm.label); }
  draw(lm);
  const disp = sm.label === "nothing" ? "–" : sm.label;
  updateUI(disp, conf, sm.frac, true, word.text);
}

function updateUI(letter, conf, stab, hand, text) {
  els.letter.textContent = letter;
  els.hand.textContent = hand ? "yes" : "no";
  els.source.textContent = hand ? "detected" : "no hand";
  els.source.className = "source-tag" + (hand ? " source-static" : "");
  const c = Math.round(conf * 100), s = Math.round(stab * 100);
  els.conf.style.width = c + "%"; els.confVal.textContent = c + "%";
  els.stab.style.width = s + "%"; els.stabVal.textContent = s + "%";
  els.word.textContent = text;
}

function draw(landmarks) {
  const w = els.overlay.width, h = els.overlay.height;
  octx.clearRect(0, 0, w, h);
  if (!landmarks) return;
  const pts = landmarks.map((p) => [p.x * w, p.y * h]);
  octx.strokeStyle = "rgba(255,255,255,.65)"; octx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) { octx.beginPath(); octx.moveTo(pts[a][0], pts[a][1]); octx.lineTo(pts[b][0], pts[b][1]); octx.stroke(); }
  octx.fillStyle = "#12c98a";
  for (const [x, y] of pts) { octx.beginPath(); octx.arc(x, y, 4, 0, Math.PI*2); octx.fill(); }
}

els.start.addEventListener("click", startCamera);
els.stop.addEventListener("click", stopCamera);
els.space.addEventListener("click", () => { word.text += " "; word.handMissing(); els.word.textContent = word.text; });
els.back.addEventListener("click", () => { word.text = word.text.slice(0, -1); word.handMissing(); els.word.textContent = word.text; });
els.clear.addEventListener("click", () => { word.reset(); els.word.textContent = ""; });

init().catch((e) => { console.error(e); setStatus("err", "Load failed — see console"); });
