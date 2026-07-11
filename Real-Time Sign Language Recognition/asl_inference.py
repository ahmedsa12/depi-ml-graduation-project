"""Reusable inference engine (STATIC-ONLY) - imported verbatim by the FastAPI backend."""
import os
import json
from collections import deque, Counter

import numpy as np
import joblib
from tensorflow.keras.models import load_model

from asl_landmarks import create_hand_detector, extract_raw_landmarks, normalize_landmarks


class SignLanguageTranslator:
    def __init__(self, models_dir="models", config=None):
        if config is None:
            with open(os.path.join(models_dir, "config.json")) as f:
                config = json.load(f)
        self.cfg = config
        self.static_model = load_model(os.path.join(models_dir, "StaticModel.keras"))
        self.static_scaler = joblib.load(os.path.join(models_dir, "StaticScaler.pkl"))
        self.static_encoder = joblib.load(os.path.join(models_dir, "StaticLabelEncoder.pkl"))
        self.min_confidence = float(config.get("min_confidence", 0.55))
        self.detector = create_hand_detector(static_image_mode=False)
        self.smooth = deque(maxlen=config["smoothing_window"])

    def reset(self):
        self.smooth.clear()

    def _static_predict(self, coords):
        feat = self.static_scaler.transform(normalize_landmarks(coords).reshape(1, -1))
        probs = self.static_model(feat, training=False).numpy()[0]
        idx = int(probs.argmax())
        return self.static_encoder.classes_[idx], float(probs[idx])

    def process_landmarks(self, coords):
        if coords is None:
            return self._finalize("nothing", 1.0, "none")
        label, conf = self._static_predict(coords)
        if conf < self.min_confidence:
            return self._finalize("nothing", conf, "static")
        return self._finalize(label, conf, "static")

    def process_frame(self, frame_bgr):
        return self.process_landmarks(extract_raw_landmarks(frame_bgr, self.detector))

    def _finalize(self, label, conf, source):
        self.smooth.append(label)
        smoothed = Counter(self.smooth).most_common(1)[0][0]
        agree = sum(1 for x in self.smooth if x == smoothed) / len(self.smooth)
        return {"letter": smoothed, "raw_letter": label, "confidence": conf,
                "stability": agree, "source": source}


class WordBuilder:
    """Builds text from stable predictions: hold a gesture -> commit once ->
    release the hand before the next character."""

    def __init__(self, stability_frames=12):
        self.stability = stability_frames
        self.reset()

    def reset(self):
        self.text = ""
        self._candidate = None
        self._count = 0
        self._locked = False

    def hand_missing(self):
        self._candidate = None
        self._count = 0
        self._locked = False

    def update(self, token):
        if token != self._candidate:
            self._candidate = token
            self._count = 1
        else:
            self._count += 1
        if self._locked:
            return self.text
        if self._count < self.stability:
            return self.text
        self._commit(token)
        self._locked = True
        return self.text

    def _commit(self, token):
        if token in ("del", "delete"):
            self.text = self.text[:-1]
        elif token == "space":
            self.text += " "
        else:
            self.text += token
