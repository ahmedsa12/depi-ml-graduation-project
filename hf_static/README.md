---
title: Real-Time Sign Language Recognition
emoji: 🖐️
colorFrom: green
colorTo: indigo
sdk: static
pinned: false
license: mit
---

# Real-Time Sign Language Recognition

Real-time American Sign Language (ASL) **fingerspelling** recognition that runs
**100% in your browser** — no server, and your webcam never leaves your device.
A static-pose **MLP** classifies the **24 static ASL letters** (`A–I, K–Y`) from
21 MediaPipe hand landmarks. Built for the **DEPI · Microsoft Machine Learning
Program**.

**Click "Start camera", allow access, and sign.** Hold a letter steady for ~1
second to type it; release your hand before repeating the same letter. Use
**Space / Backspace / Clear** to edit the text.

> The motion letters `J` and `Z` require movement and are out of scope.

## How it works

`webcam → MediaPipe HandLandmarker (WASM, 21 pts) → normalize (63-D) →
StandardScaler → MLP (in JavaScript) → confidence gate → temporal smoothing →
letter → word`

The MLP weights (`Dense 256→128→64→24` with BatchNorm) were exported from the
trained Keras model to `weights.json` and re-implemented in plain JavaScript, so
inference runs locally with no backend.

**Team:** Ahmed Abd El-Ghafar Slama · Ahmed Hassan Mohamed ·
Yara Ahmed Mohamed Abdelall · Salwa Wael Mohammed Elkordy
