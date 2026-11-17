"""Deterministic acoustic features. No pretrained or hidden learned encoders."""
import subprocess
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.signal import resample_poly

SAMPLE_RATE = 16000
FEATURE_VERSION = "mfcc-prosody-v1"


def decode_audio(path: str | Path) -> np.ndarray:
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-t", "60", "-f", "f32le", "pipe:1"],
        capture_output=True, timeout=30, check=True,
    )
    audio = np.frombuffer(result.stdout, dtype="<f4").copy()
    if audio.size < 160:
        raise ValueError("Audio must contain at least 10 ms of decodable sound")
    return audio


def features(audio: np.ndarray) -> np.ndarray:
    """MFCC mean/std plus spectral, energy, zero crossing and pitch summaries."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1 or len(audio) < 160 or not np.isfinite(audio).all():
        raise ValueError("Expected finite mono 16 kHz audio, at least 10 ms")
    duration = len(audio) / SAMPLE_RATE
    audio = np.pad(audio, (0, max(0, 400 - len(audio))))
    frames = np.lib.stride_tricks.sliding_window_view(audio, 400)[::160].copy()
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    power = np.abs(np.fft.rfft(frames * np.hanning(400), n=512)) ** 2
    frequencies = np.fft.rfftfreq(512, 1 / SAMPLE_RATE)
    mel_edges = 700 * (10 ** (np.linspace(0, 2595 * np.log10(1 + 8000 / 700), 28) / 2595) - 1)
    filters = []
    for left, center, right in zip(mel_edges[:-2], mel_edges[1:-1], mel_edges[2:]):
        filters.append(np.maximum(0, np.minimum((frequencies-left)/(center-left), (right-frequencies)/(right-center))))
    mfcc = dct(np.log(power @ np.array(filters).T + 1e-10), type=2, norm="ortho", axis=1)[:, :13]
    centroid = (power @ frequencies) / (power.sum(axis=1) + 1e-10) / 8000
    flatness = np.exp(np.mean(np.log(power + 1e-10), axis=1)) / (np.mean(power, axis=1) + 1e-10)
    # 8 kHz autocorrelation estimates pitch from 80 to 400 Hz.
    low = resample_poly(audio, 1, 2)
    low = np.pad(low, (0, max(0, 320-len(low))))
    pitch_frames = np.lib.stride_tricks.sliding_window_view(low, 320)[::160].copy()
    pitch_frames -= pitch_frames.mean(axis=1, keepdims=True)
    spectrum = np.fft.rfft(pitch_frames * np.hanning(320), n=1024)
    correlation = np.fft.irfft(np.abs(spectrum)**2, n=1024)[:, :101]
    lags = np.argmax(correlation[:, 20:101], axis=1) + 20
    strength = correlation[np.arange(len(lags)), lags] / (correlation[:, 0]+1e-10)
    voiced = strength > .35
    pitches = 8000 / lags[voiced]
    pitch_summary = [float(pitches.mean())/400 if pitches.size else 0,
                     float(pitches.std())/400 if pitches.size else 0, float(voiced.mean())]
    summaries = [mfcc.mean(axis=0), mfcc.std(axis=0)]
    for values in (np.log(rms+1e-8), zcr, centroid, flatness):
        summaries.append(np.array([values.mean(), values.std(), np.percentile(values, 90)-np.percentile(values, 10)]))
    return np.concatenate([*summaries, [duration], pitch_summary]).astype(np.float32)
