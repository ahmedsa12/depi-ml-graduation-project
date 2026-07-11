"""
Real-Time Sign Language Recognition — FastAPI backend.

This service is a THIN wrapper around the trained model bundle. It imports the
exact `asl_inference.py` / `asl_landmarks.py` modules produced by the notebook —
the machine-learning code is never re-implemented here.

Endpoints
---------
GET  /                 -> service metadata
GET  /health           -> liveness + model status
GET  /config           -> feature dims, classes, thresholds (from config.json)
POST /predict          -> classify one webcam frame (base64 image or file upload)
POST /predict/landmarks-> classify from client-side landmarks (browser MediaPipe)
POST /word/edit        -> manual space / backspace / clear on the current word
POST /reset            -> clear a session's smoothing + word
WS   /ws               -> stream frames and receive predictions (low latency)

Static-only: a single static-pose MLP classifies the 24 letters (A–I, K–Y).
Each client is identified by `session_id` and gets its OWN smoothing window and
word-builder state, while sharing the (heavy) loaded model.

Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import os
import sys
import copy
import time
import base64
import threading
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Locate the trained model bundle and import the project's ML modules verbatim.
# ---------------------------------------------------------------------------
def _find_models_dir() -> str:
    env = os.environ.get("MODELS_DIR")
    candidates = [env] if env else []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates += [
        os.path.join(here, "..", "Real-Time Sign Language Recognition"),
        os.path.join(here, "..", "models"),
        os.path.join(here, "models"),
        os.path.join(here, ".."),
    ]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "asl_inference.py")):
            return os.path.abspath(c)
    raise RuntimeError(
        "Could not find the model bundle (asl_inference.py). "
        "Set the MODELS_DIR environment variable to the folder that contains "
        "StaticModel.keras, StaticScaler.pkl, asl_inference.py, etc."
    )


MODELS_DIR = _find_models_dir()
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

# These are the project's own modules — imported, not re-implemented.
from asl_inference import SignLanguageTranslator, WordBuilder      # noqa: E402
from asl_landmarks import create_hand_detector, extract_raw_landmarks  # noqa: E402


# ---------------------------------------------------------------------------
# Session management: one shared set of models, per-client runtime state.
# ---------------------------------------------------------------------------
class Session:
    def __init__(self, translator: SignLanguageTranslator, word: WordBuilder):
        self.translator = translator
        self.word = word
        self.lock = threading.Lock()
        self.last_used = time.time()


class Engine:
    """Loads the models once and hands out lightweight per-session clones."""

    def __init__(self, models_dir: str):
        self.base = SignLanguageTranslator(models_dir)   # loads models ONCE
        self.cfg = self.base.cfg
        self.sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _clone_translator(self) -> SignLanguageTranslator:
        # Share the heavy Keras model / scaler / encoder; give this session its
        # own detector + smoothing window so clients never interfere.
        t = copy.copy(self.base)
        t.detector = create_hand_detector(static_image_mode=False)
        t.smooth = deque(maxlen=self.cfg["smoothing_window"])
        return t

    def session(self, session_id: str = "default") -> Session:
        with self._lock:
            s = self.sessions.get(session_id)
            if s is None:
                translator = self.base if session_id == "default" else self._clone_translator()
                s = Session(translator, WordBuilder(self.cfg["stability_frames"]))
                self.sessions[session_id] = s
            s.last_used = time.time()
            return s

    def reset(self, session_id: str = "default"):
        s = self.session(session_id)
        with s.lock:
            s.translator.reset()
            s.word.reset()


ENGINE: Engine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ENGINE
    print(f"[startup] loading models from: {MODELS_DIR}")
    ENGINE = Engine(MODELS_DIR)
    print(f"[startup] ready (static-only). classes={ENGINE.cfg['static_classes']}")
    yield
    print("[shutdown] bye")


app = FastAPI(
    title="Real-Time Sign Language Recognition API",
    description="Landmark-based ASL alphabet recognition (static-pose MLP, 24 letters).",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow the frontend (any origin during development) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_base64_image(data: str) -> np.ndarray:
    """Decode a base64 (optionally data-URL) string into a BGR frame."""
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", 1)[1]
    raw = base64.b64decode(data)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image data.")
    return frame


def _run_on_frame(session: Session, frame_bgr: np.ndarray) -> dict:
    """Extract landmarks once, route through the model, update the word."""
    with session.lock:
        coords = extract_raw_landmarks(frame_bgr, session.translator.detector)
        result = session.translator.process_landmarks(coords)
        letter = result["letter"]
        if coords is None or letter == "nothing":
            session.word.hand_missing()
        else:
            session.word.update(letter)
        result["word"] = session.word.text
        result["hand_detected"] = coords is not None
        result["landmarks"] = (
            np.round(coords[:, :2], 4).tolist() if coords is not None else None
        )
        return result


def _run_on_landmarks(session: Session, coords: np.ndarray | None) -> dict:
    with session.lock:
        result = session.translator.process_landmarks(coords)
        letter = result["letter"]
        if coords is None or letter == "nothing":
            session.word.hand_missing()
        else:
            session.word.update(letter)
        result["word"] = session.word.text
        result["hand_detected"] = coords is not None
        return result


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class PredictBody(BaseModel):
    image: str                      # base64 or data-URL JPEG/PNG
    session_id: str = "default"


class LandmarksBody(BaseModel):
    landmarks: list                 # 21x3 nested OR flat 63
    session_id: str = "default"


class WordEditBody(BaseModel):
    action: str                     # "space" | "backspace" | "clear"
    session_id: str = "default"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "service": "Real-Time Sign Language Recognition API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/health", "/config", "/predict", "/predict/landmarks",
                      "/word/edit", "/reset", "/ws"],
    }


@app.get("/health")
def health():
    ready = ENGINE is not None
    return {
        "status": "ok" if ready else "loading",
        "models_dir": MODELS_DIR,
        "mode": "static-only",
        "static_classes": ENGINE.cfg["static_classes"] if ready else [],
    }


@app.get("/config")
def config():
    c = ENGINE.cfg
    return {
        "mode": "static-only",
        "feature_dim": c["feature_dim"],
        "static_classes": c["static_classes"],
        "smoothing_window": c["smoothing_window"],
        "stability_frames": c["stability_frames"],
        "min_confidence": c.get("min_confidence", 0.55),
    }


@app.post("/predict")
def predict(body: PredictBody):
    """Classify a single frame from a JSON body {image, session_id}."""
    frame = _decode_base64_image(body.image)
    return _run_on_frame(ENGINE.session(body.session_id), frame)


@app.post("/predict/file")
async def predict_file(
    file: UploadFile = File(...),
    session_id: str = Form(default="default"),
):
    """Classify a single frame from a multipart/form-data file upload."""
    raw = np.frombuffer(await file.read(), dtype=np.uint8)
    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode uploaded image.")
    return _run_on_frame(ENGINE.session(session_id), frame)


@app.post("/predict/landmarks")
def predict_landmarks(body: LandmarksBody):
    """Classify from client-supplied landmarks (browser-side MediaPipe path)."""
    arr = np.asarray(body.landmarks, dtype=np.float32)
    if arr.size == 63:
        coords = arr.reshape(21, 3)
    elif arr.shape == (21, 3):
        coords = arr
    elif arr.size == 0:
        coords = None
    else:
        raise HTTPException(status_code=400, detail="landmarks must be 21x3 or flat 63.")
    return _run_on_landmarks(ENGINE.session(body.session_id), coords)


@app.post("/word/edit")
def word_edit(body: WordEditBody):
    """Manual text editing (space / backspace / clear) — the alphabet model does
    not emit space/delete gestures, so the UI drives these directly."""
    s = ENGINE.session(body.session_id)
    with s.lock:
        if body.action == "space":
            s.word.text += " "
        elif body.action == "backspace":
            s.word.text = s.word.text[:-1]
        elif body.action == "clear":
            s.word.reset()
        else:
            raise HTTPException(status_code=400, detail="action must be space|backspace|clear.")
        return {"word": s.word.text}


@app.post("/reset")
def reset(session_id: str = "default"):
    ENGINE.reset(session_id)
    return {"status": "reset", "session_id": session_id}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Stream loop: client sends {image} JSON, server replies with the prediction."""
    await websocket.accept()
    sid = websocket.query_params.get("session_id", "default")
    session = ENGINE.session(sid)
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "reset":
                ENGINE.reset(sid)
                await websocket.send_json({"word": ""})
                continue
            frame = _decode_base64_image(msg["image"])
            result = _run_on_frame(session, frame)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        return
    except Exception as e:  # keep the socket error explicit for the client
        await websocket.send_json({"error": str(e)})


@app.exception_handler(Exception)
async def _unhandled(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})
