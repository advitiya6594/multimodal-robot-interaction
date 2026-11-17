"""Per-turn local inference and an explicit, deterministic response policy."""
from pathlib import Path
import time

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .audio import decode_audio, features, FEATURE_VERSION
from .data import ROOT
from .train import fuse

POLICY_THRESHOLD = 0.4
REPLIES = {
    "anger": "That sounds frustrating. What would help right now?",
    "disgust": "It sounds like that bothered you. Want to tell me more?",
    "fear": "That sounds worrying. What would help you feel more comfortable?",
    "joy": "That sounds exciting. Tell me more!",
    "neutral": "I'm listening. What happened next?",
    "sadness": "That sounds difficult. Would you like to talk about it?",
    "surprise": "That sounds unexpected. How do you feel about it?",
}
BEHAVIORS = {
    "anger": ("warm", "slow", "small_nod"), "disgust": ("warm", "slow", "attentive"),
    "fear": ("warm", "slow", "attentive"), "joy": ("gold", "bright", "gentle_bounce"),
    "neutral": ("white", "steady", "attentive"), "sadness": ("warm", "slow", "small_nod"),
    "surprise": ("gold", "steady", "head_tilt"),
}


class Engine:
    def __init__(self, model_path: Path = ROOT / "artifacts/model.joblib"):
        # Only load the artifact distributed with this repository or one you trained.
        self.bundle = joblib.load(model_path)
        if self.bundle["feature_version"] != FEATURE_VERSION:
            raise ValueError("Feature version mismatch. Retrain the model.")

    def predict(self, text: str, audio_path=None, mode="fusion", turn_id=None):
        if mode not in ("text", "audio", "fusion"):
            raise ValueError("Mode must be text, audio or fusion")
        if not isinstance(text, str) or not text.strip() or len(text) > 4000:
            raise ValueError("Supply a transcript of 1–4000 characters")
        if mode != "text" and audio_path is None:
            raise ValueError("Audio is required for audio and fusion modes")
        started = time.perf_counter()
        with threadpool_limits(limits=1):
            acoustic = None
            if mode != "text":
                acoustic = self.bundle["scaler"].transform(features(decode_audio(audio_path))[None, :])
            words = self.bundle["vectorizer"].transform([text])
            inputs = {"text": words, "audio": acoustic}
            if mode == "fusion":
                inputs["fusion"] = fuse(words, acoustic, self.bundle["audio_weight"])
            model = self.bundle[mode]
            probabilities = model.predict_proba(inputs[mode])[0]
        best = int(np.argmax(probabilities))
        emotion, confidence = str(model.classes_[best]), float(probabilities[best])
        uncertain = confidence < POLICY_THRESHOLD
        quote = " ".join(text.split())[:90]
        response = f'You said, “{quote}”. ' + ("How are you feeling about that?" if uncertain else REPLIES[emotion])
        color, tempo, motion = ("white", "steady", "attentive") if uncertain else BEHAVIORS[emotion]
        return {
            "turn_id": turn_id, "emotion": emotion, "confidence": round(confidence, 4),
            "scores": {str(c): round(float(p), 4) for c, p in zip(model.classes_, probabilities)},
            "uncertain": uncertain, "response": response,
            "behavior": {"light": color, "tempo": tempo, "motion": motion},
            "modalities": ["text", "audio"] if mode == "fusion" else [mode],
            "latency_ms": round((time.perf_counter()-started)*1000, 2),
            "policy": "templates-v1", "confidence_note": "Uncalibrated model score. Threshold 0.4 is a fixed interaction heuristic.",
        }
