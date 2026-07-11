/* Real-Time Sign Language Recognition — frontend client
   Streams downscaled webcam frames to the backend over WebSocket and renders
   the returned letter / confidence / word plus the hand-landmark overlay. */

const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), overlay: $("overlay"), hint: $("videoHint"),
  letter: $("letter"), source: $("sourceTag"),
  conf: $("confBar"), confVal: $("confVal"),
  stab: $("stabBar"), stabVal: $("stabVal"),
  fps: $("fps"), hand: $("hand"), word: $("word"),
  statusDot: $("statusDot"), statusText: $("statusText"),
  start: $("startBtn"), stop: $("stopBtn"),
  space: $("spaceBtn"), back: $("backBtn"), clear: $("clearBtn"),
  url: $("backendUrl"), size: $("sendSize"),
};

const SESSION_ID = "web-" + Math.random().toString(36).slice(2, 10);
const send = document.createElement("canvas");    // offscreen capture buffer
const sctx = send.getContext("2d");
const octx = els.overlay.getContext("2d");

let stream = null, ws = null, running = false, inFlight = false;
let lastTs = 0, targetInterval = 70; // ~14 fps ceiling

// ---------- status helpers ----------
function setStatus(state, text) {
  els.statusDot.className = "dot " + ({ on: "dot-on", off: "dot-off", err: "dot-err" }[state]);
  els.statusText.textContent = text;
}
function backendBase() { return els.url.value.trim().replace(/\/+$/, ""); }
function wsUrl() {
  const b = backendBase().replace(/^http/, "ws");
  return `${b}/ws?session_id=${encodeURIComponent(SESSION_ID)}`;
}

// ---------- camera ----------
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
    els.video.srcObject = stream;
    await els.video.play();
    els.hint.style.display = "none";
    sizeOverlay();
    running = true;
    els.start.disabled = true; els.stop.disabled = false;
    [els.space, els.back, els.clear].forEach((b) => (b.disabled = false));
    await fetch(`${backendBase()}/reset?session_id=${SESSION_ID}`, { method: "POST" }).catch(() => {});
    connectWs();
  } catch (e) {
    setStatus("err", "Camera blocked");
    alert("Could not access the webcam: " + e.message);
  }
}

function stopCamera() {
  running = false;
  if (ws) { ws.close(); ws = null; }
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  els.start.disabled = false; els.stop.disabled = true;
  [els.space, els.back, els.clear].forEach((b) => (b.disabled = true));
  octx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.hint.style.display = "grid";
  setStatus("off", "Disconnected");
  els.letter.textContent = "–";
  els.source.textContent = "idle"; els.source.className = "source-tag";
}

function sizeOverlay() {
  els.overlay.width = els.video.clientWidth;
  els.overlay.height = els.video.clientHeight;
}
window.addEventListener("resize", () => running && sizeOverlay());

// ---------- websocket loop ----------
function connectWs() {
  setStatus("off", "Connecting…");
  ws = new WebSocket(wsUrl());
  ws.onopen = () => { setStatus("on", "Live"); inFlight = false; pump(); };
  ws.onmessage = (ev) => {
    inFlight = false;
    const data = JSON.parse(ev.data);
    if (data.error) { setStatus("err", "Server error"); console.error(data.error); }
    else render(data);
    const wait = Math.max(0, targetInterval - (performance.now() - lastTs));
    setTimeout(pump, wait);
  };
  ws.onerror = () => setStatus("err", "Connection error");
  ws.onclose = () => { if (running) setStatus("err", "Disconnected"); };
}

function pump() {
  if (!running || !ws || ws.readyState !== WebSocket.OPEN || inFlight) return;
  const w = Math.max(160, Math.min(640, parseInt(els.size.value) || 320));
  const h = Math.round((w * 3) / 4);
  send.width = w; send.height = h;
  sctx.drawImage(els.video, 0, 0, w, h);
  const image = send.toDataURL("image/jpeg", 0.6);
  lastTs = performance.now();
  inFlight = true;
  ws.send(JSON.stringify({ image }));
}

// ---------- rendering ----------
let fpsEma = 0, lastFrame = 0;
function render(d) {
  const now = performance.now();
  if (lastFrame) {
    const inst = 1000 / (now - lastFrame);
    fpsEma = fpsEma ? fpsEma * 0.8 + inst * 0.2 : inst;
    els.fps.textContent = fpsEma.toFixed(0);
  }
  lastFrame = now;

  const letter = d.letter === "nothing" ? "–" : d.letter;
  els.letter.textContent = letter;
  els.hand.textContent = d.hand_detected ? "yes" : "no";

  els.source.textContent = d.hand_detected ? "detected" : "no hand";
  els.source.className = "source-tag" + (d.hand_detected ? " source-static" : "");

  const conf = Math.round((d.confidence || 0) * 100);
  const stab = Math.round((d.stability || 0) * 100);
  els.conf.style.width = conf + "%"; els.confVal.textContent = conf + "%";
  els.stab.style.width = stab + "%"; els.stabVal.textContent = stab + "%";

  els.word.textContent = d.word || "";
  drawLandmarks(d.landmarks, d.source);
}

function drawLandmarks(pts, source) {
  const w = els.overlay.width, h = els.overlay.height;
  octx.clearRect(0, 0, w, h);
  if (!pts || !pts.length) return;
  const color = "#12c98a";
  octx.strokeStyle = "rgba(255,255,255,.65)";
  octx.lineWidth = 2;
  for (const [a, b] of HAND_CONNECTIONS) {
    octx.beginPath();
    octx.moveTo(pts[a][0] * w, pts[a][1] * h);
    octx.lineTo(pts[b][0] * w, pts[b][1] * h);
    octx.stroke();
  }
  octx.fillStyle = color;
  for (const [x, y] of pts) {
    octx.beginPath();
    octx.arc(x * w, y * h, 4, 0, Math.PI * 2);
    octx.fill();
  }
}

// ---------- word editing ----------
async function wordEdit(action) {
  try {
    const r = await fetch(`${backendBase()}/word/edit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action, session_id: SESSION_ID }),
    });
    const d = await r.json();
    els.word.textContent = d.word || "";
  } catch (e) { console.error(e); }
}

// ---------- wire up ----------
els.start.addEventListener("click", startCamera);
els.stop.addEventListener("click", stopCamera);
els.space.addEventListener("click", () => wordEdit("space"));
els.back.addEventListener("click", () => wordEdit("backspace"));
els.clear.addEventListener("click", () => wordEdit("clear"));
setStatus("off", "Disconnected");
