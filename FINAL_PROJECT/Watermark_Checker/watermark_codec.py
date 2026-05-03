from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


@dataclass
class DecodeResult:
    payload_id: int
    bit_scores: list[float]
    bit_confidence: list[float]

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.bit_confidence)) if self.bit_confidence else 0.0

    @property
    def bit_string(self) -> str:
        bits = ["1" if score > 0 else "0" for score in self.bit_scores]
        return "".join(reversed(bits))


def key_to_seed(key: str) -> int:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return struct.unpack("<Q", digest[:8])[0]


def _pn_matrix(secret_key: str, n_frames: int, n_bins: int) -> np.ndarray:
    rng = np.random.default_rng(key_to_seed(secret_key))
    return rng.choice(np.array([-1.0, 1.0]), size=(n_frames, n_bins)).astype(np.float32)


def _mono_float(audio: np.ndarray) -> np.ndarray:
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform


def _magnitude(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    _, _, zxx = stft(
        audio,
        fs=sample_rate,
        nperseg=1024,
        noverlap=768,
        boundary="zeros",
        padded=True,
    )
    return np.abs(zxx).astype(np.float32)


def _decode_from_feature(
    feature: np.ndarray,
    secret_key: str,
    bits: int,
    bin_offset: int = 0,
    total_bins: int | None = None,
) -> DecodeResult:
    n_bins, n_frames = feature.shape
    carrier_bins = total_bins if total_bins is not None else n_bins
    pn = _pn_matrix(secret_key, n_frames, carrier_bins).T[bin_offset : bin_offset + n_bins]

    bit_scores: list[float] = []
    bit_confidence: list[float] = []

    for bit_index in range(bits):
        frame_mask = (np.arange(n_frames) % bits) == bit_index
        if not np.any(frame_mask):
            bit_scores.append(0.0)
            bit_confidence.append(0.0)
            continue

        corr_values = feature[:, frame_mask] * pn[:, frame_mask]
        score = float(np.mean(corr_values))
        standard_error = float(np.std(corr_values) / np.sqrt(corr_values.size) + 1e-12)
        confidence = min(1.0, abs(score) / standard_error / 8.0)
        bit_scores.append(score)
        bit_confidence.append(confidence)

    payload_id = 0
    for index, score in enumerate(bit_scores):
        if score > 0:
            payload_id += 1 << index

    return DecodeResult(payload_id=payload_id, bit_scores=bit_scores, bit_confidence=bit_confidence)


def decode_payload(audio: np.ndarray, sample_rate: int, secret_key: str, bits: int = 32) -> DecodeResult:
    """Blind decode STFT magnitude-domain spread-spectrum payload.

    This matches Nick's OutputProcessor/FramewiseWatermarker layout: each STFT
    frame carries one BPSK payload bit through a keyed pseudo-random carrier.
    """
    waveform = _mono_float(audio)
    if waveform.size == 0:
        return DecodeResult(payload_id=0, bit_scores=[0.0] * bits, bit_confidence=[0.0] * bits)

    magnitude = _magnitude(waveform, sample_rate)
    total_bins = magnitude.shape[0]
    high_bin_start = max(1, total_bins // 4)
    magnitude = magnitude[high_bin_start:, :]

    frame_rms = np.sqrt(np.mean(magnitude**2, axis=0, keepdims=True) + 1e-8)
    return _decode_from_feature(
        magnitude / frame_rms,
        secret_key,
        bits,
        bin_offset=high_bin_start,
        total_bins=total_bins,
    )
