<h1 align="center">Real-Time Sign Language Recognition</h1>

<p align="center">
  Real-time <b>American Sign Language (ASL) fingerspelling</b> recognition from a webcam —
  landmark-based, runs <b>100% in the browser</b>, and built as the graduation project for the
  <b>DEPI · Microsoft Machine Learning Program</b>.
</p>

<p align="center">
  <a href="https://huggingface.co/spaces/salam0z/Real-Time-Sign-Language-Recognition">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Beta-047954?style=for-the-badge" alt="Live Demo"></a>
</p>

<p align="center">
  <a href="https://www.python.org"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://www.tensorflow.org"><img src="https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow"></a>
  <a href="https://ai.google.dev/edge/mediapipe"><img src="https://img.shields.io/badge/MediaPipe-0.10.35-0097A7?logo=google&logoColor=white" alt="MediaPipe"></a>
  <a href="https://scikit-learn.org"><img src="https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?logo=scikitlearn&logoColor=white" alt="scikit-learn"></a>
  <a href="https://numpy.org"><img src="https://img.shields.io/badge/NumPy-1.26%2B-013243?logo=numpy&logoColor=white" alt="NumPy"></a>
  <a href="https://opencv.org"><img src="https://img.shields.io/badge/OpenCV-4.9%2B-5C3EE8?logo=opencv&logoColor=white" alt="OpenCV"></a>
  <a href="https://fastapi.tiangolo.com"><img src="https://img.shields.io/badge/FastAPI-0.111%2B-009688?logo=fastapi&logoColor=white" alt="FastAPI"></a>
  <a href="https://www.uvicorn.org"><img src="https://img.shields.io/badge/Uvicorn-0.30%2B-4B8BBE?logo=uvicorn&logoColor=white" alt="Uvicorn"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-3DA639" alt="License"></a>
</p>

> **Try it now (Beta):** <https://huggingface.co/spaces/salam0z/Real-Time-Sign-Language-Recognition>
> — open the page, click **Start camera**, and sign. Everything runs on your device; no video is uploaded.

---

## Table of Contents
- [Overview](#-overview)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Repository Structure](#-repository-structure)
- [Model & Classes](#-model--classes)
- [Getting Started](#-getting-started)
- [API Documentation](#-api-documentation)
- [Results](#-results)
- [Team](#-team)
- [Limitations](#-limitations)
- [Acknowledgments](#-acknowledgments)

---

## Overview

Instead of classifying raw pixels, the system detects **21 hand landmarks** with MediaPipe, turns
them into a compact **63-D geometric feature vector**, normalises it to be **translation- and
scale-invariant**, and classifies it with a small **MLP**. This makes the model tiny, fast on CPU,
and robust to background, lighting, camera, and skin tone.

- **98.74% accuracy** · **0.986 macro F1** on 11,074 held-out samples.
- Recognises the **24 static ASL letters** (`A–I, K–Y`).
- **Real-time** on CPU; a confidence gate + temporal smoothing keep predictions stable.
- **Deploys anywhere** — Colab notebook, FastAPI API, and a **fully in-browser** static app.
- **Private** — the browser version runs entirely on-device; the webcam never leaves the machine.

> The motion letters **J** and **Z** require movement and are intentionally out of scope; every
> prediction is a stable single-frame pose.

---

## Tech Stack

| Category | Technology | Version |
|---|---|---|
| **Language** | Python | 3.10 – 3.12 |
| **Deep learning** | TensorFlow / Keras | ≥ 2.16 |
| **Hand tracking** | MediaPipe (Tasks HandLandmarker) | 0.10.35 |
| **Classic ML** | scikit-learn | ≥ 1.3 |
| **Numerics** | NumPy | ≥ 1.26 |
| **Vision I/O** | OpenCV (headless) | ≥ 4.9 |
| **Serialization** | joblib | ≥ 1.3 |
| **Protobuf** | protobuf | ≥ 5.28 |
| **Backend** | FastAPI | ≥ 0.111 |
| **ASGI server** | Uvicorn | ≥ 0.30 |
| **Validation** | Pydantic | ≥ 2.7 |
| **Multipart** | python-multipart | ≥ 0.0.9 |
| **Frontend** | HTML · CSS · vanilla JS · WebSocket · Canvas | — |
| **In-browser ML** | MediaPipe Tasks (WASM) + hand-written JS MLP | — |
| **Tooling** | Google Colab · Hugging Face Spaces · Git | — |

Install the Python dependencies with:

```bash
pip install -r backend/requirements.txt
```

---

## Architecture

```
 Webcam frame
      │
      ▼
 MediaPipe HandLandmarker ──► 21 (x, y, z) landmarks
      │
      ▼
 Normalization  (wrist-relative + scale-invariant)  ──► 63-D vector
      │
      ▼
 StandardScaler ──► Static MLP  (Dense 256→128→64 → Softmax, 24 classes)
      │
      ▼
 confidence gate → temporal smoothing (majority vote) → letter → Word Builder → text
```

---

## Repository Structure

```
.
├── Real-Time Sign Language Recognition.ipynb   # end-to-end training notebook (Colab)
├── Real-Time Sign Language Recognition/        # trained model bundle
│   ├── StaticModel.keras                       # the single trained model
│   ├── StaticScaler.pkl   StaticLabelEncoder.pkl
│   ├── asl_landmarks.py   asl_inference.py      # reusable ML code (imported verbatim)
│   ├── hand_landmarker.task
│   └── class_mapping.json  config.json
├── backend/          # FastAPI service (REST + WebSocket)
│   ├── app.py  requirements.txt  run.sh  run.bat
├── frontend/         # zero-build web client (webcam → live translation)
│   ├── index.html  styles.css  app.js
├── hf_static/        # the live in-browser app (deployed on Hugging Face)
│   ├── index.html  app.js  styles.css  weights.json  hand_landmarker.task
├── docs/             # API.md · BUSINESS_IMPACT.md
├── documentation/    # project report, explanation book, study guide (PDF + Word)
└── presentation/     # DEPI defense deck
```

---

## Model & Classes

- **Model:** a single MLP — `63 → 256 → 128 → 64 → 24` with BatchNorm, ReLU, Dropout, and a Softmax head.
- **Static letters (24):** `A B C D E F G H I K L M N O P Q R S T U V W X Y`
- **Not modelled:** `J`, `Z` (motion letters) and `space` / `delete` / `nothing`.
- **Input:** 21 landmarks × (x, y, z) = **63** features.
- **Dataset:** [ASL Alphabet (Kaggle)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — ~87k images, ~3,000/class, balanced.

---

## Getting Started

### 1) Try the live demo (nothing to install)
Open **<https://huggingface.co/spaces/salam0z/Real-Time-Sign-Language-Recognition>** → **Start camera**.

### 2) Reproduce / retrain (optional)
Open **`Real-Time Sign Language Recognition.ipynb`** in Google Colab and *Run all*. It downloads the
dataset, extracts landmarks, trains the model, evaluates it, and exports the model bundle.

### 3) Run the backend API
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
./run.sh                                                # Windows: run.bat
# API on http://localhost:8000  ·  interactive docs at /docs
```

### 4) Run the web frontend
```bash
cd frontend
python -m http.server 5500
# open http://localhost:5500  → Start camera
```

---

## API Documentation

**Base URL (dev):** `http://localhost:8000` · **Interactive docs:** `/docs` (Swagger) · `/redoc`
Static-only: a single MLP classifies the 24 letters. State (smoothing window + current word) is kept
per `session_id` (omit it to use `"default"`).

### Prediction result object
Returned by every prediction endpoint:

| Field | Type | Description |
|---|---|---|
| `letter` | string | Smoothed prediction; `"nothing"` when no hand or below the confidence gate. |
| `raw_letter` | string | This frame's raw prediction (before smoothing). |
| `confidence` | float | Top-class probability, `0.0–1.0`. |
| `stability` | float | Fraction of the smoothing window agreeing with `letter`. |
| `source` | string | `"static"` or `"none"` (no hand). |
| `word` | string | Current translated text for the session. |
| `hand_detected` | bool | Whether a hand was found. |
| `landmarks` | array \| null | `21 × [x, y]` normalized coordinates for overlay. |

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + model status |
| `GET` | `/config` | Feature dim, classes, thresholds |
| `POST` | `/predict` | Classify one frame (JSON base64) |
| `POST` | `/predict/file` | Classify one frame (file upload) |
| `POST` | `/predict/landmarks` | Classify from client-side landmarks |
| `POST` | `/word/edit` | `space` / `backspace` / `clear` |
| `POST` | `/reset` | Clear a session |
| `WS` | `/ws` | Stream frames → predictions |

<details>
<summary><b>POST /predict</b> — classify one frame (JSON)</summary>

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"image":"data:image/jpeg;base64,/9j/4AAQ...","session_id":"demo"}'
```
```json
{ "letter": "A", "raw_letter": "A", "confidence": 0.97, "stability": 0.9,
  "source": "static", "word": "CAB", "hand_detected": true, "landmarks": [[0.51,0.62], "..."] }
```
</details>

<details>
<summary><b>POST /predict/file</b> — file upload (curl/Postman friendly)</summary>

```bash
curl -X POST http://localhost:8000/predict/file -F "file=@frame.jpg" -F "session_id=demo"
```
</details>

<details>
<summary><b>POST /predict/landmarks</b> — from client-side landmarks</summary>

```bash
curl -X POST http://localhost:8000/predict/landmarks \
  -H "Content-Type: application/json" \
  -d '{"landmarks": [[0.5,0.6,0.0], "... 21 points ..."], "session_id":"demo"}'
```
`landmarks` may be `21×3` nested or a flat `63`; empty `[]` means no hand.
</details>

<details>
<summary><b>POST /word/edit</b> · <b>POST /reset</b></summary>

```bash
curl -X POST http://localhost:8000/word/edit \
  -H "Content-Type: application/json" -d '{"action":"space","session_id":"demo"}'
curl -X POST "http://localhost:8000/reset?session_id=demo"
```
`action` is `space` | `backspace` | `clear`.
</details>

<details>
<summary><b>WS /ws</b> — streaming</summary>

Connect to `ws://localhost:8000/ws?session_id=demo`. Send `{ "image": "data:image/jpeg;base64,..." }`
per frame (or `{ "type": "reset" }`); receive a prediction object per frame. Use a request→response
cadence to self-pace to the server.
</details>

Full reference: [`docs/API.md`](docs/API.md).

---

## Results

Evaluated on a stratified held-out test set of **11,074** landmark samples.

| Metric | Score |
|---|---|
| Accuracy | **0.9874** |
| Precision (weighted) | 0.9876 |
| Recall (weighted) | 0.9874 |
| F1-score (weighted) | 0.9874 |
| F1-score (macro) | **0.9859** |

The confusion matrix is almost perfectly diagonal; the only notable confusion is **N ↔ M** (both
closed-fist poses). Figures and the full per-class report are in [`documentation/`](documentation).

---

## Team

| Member | Role |
|---|---|
| **Ahmed Abd El-Ghafar Salama** | Model Architecture · Training · Data Preprocessing |
| **Ahmed Hassan Mohamed** | Hyperparameter Tuning · Feature Engineering · Model Deployment |
| **Yara Ahmed Mohamed Abdelall** | Data Augmentation · Model Testing · Frontend Integration |
| **Salwa Wael Mohammed Elkordy** | Dataset Preparation · Model Evaluation · Performance Analysis |

---

## Limitations

- Scope is the **static ASL alphabet** (24 letters), not `J`/`Z` motion or full sign-language grammar.
- The dataset has limited signer diversity, so real-world accuracy for new users/cameras may be lower.
- An **assistive aid**, not a replacement for professional interpreters in high-stakes settings.

---

## Acknowledgments

DEPI (Digital Egypt Pioneers Initiative) & the Microsoft Machine Learning Program · EYouth ·
the Kaggle *ASL Alphabet* dataset · Google MediaPipe · Hugging Face.

<p align="center"><sub>MIT License · DEPI Graduation Project 2026</sub></p>
