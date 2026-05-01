from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class DecodeResult:
    """Structured result returned by the payload decoder."""

    payload_id: int
    bit_scores: list[float]
    bit_confidence: list[float]

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.bit_confidence)) if self.bit_confidence else 0.0

    @property
    def bit_string(self) -> str:
        """Return the decoded bits in human-readable most-significant-first order."""
        bits = ["1" if score > 0 else "0" for score in self.bit_scores]
        return "".join(reversed(bits))


def key_to_seed(key: str) -> int:
    """Convert a text key into a stable integer seed."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return struct.unpack("<Q", digest[:8])[0]


def carrier(secret_key: str, frames: int, bins: int) -> np.ndarray:
    """Legacy spread-spectrum carrier helper kept for older experiments."""
    rng = np.random.default_rng(key_to_seed(secret_key))
    return rng.choice(np.array([-1.0, 1.0]), size=(frames, bins)).astype(np.float32)


def bit_carrier(secret_key: str, bit_index: int, frames: int, bins: int) -> np.ndarray:
    """Legacy per-bit carrier helper kept for older experiments."""
    seed = key_to_seed(secret_key) ^ ((bit_index + 1) * 0x9E3779B97F4A7C15)
    rng = np.random.default_rng(seed & ((1 << 64) - 1))
    return rng.choice(np.array([-1.0, 1.0]), size=(bins, frames)).astype(np.float32)


def decode_payload(
    audio: np.ndarray,
    sample_rate: int,
    secret_key: str,
    bits: int = 32,
) -> DecodeResult:
    """Decode the fsk-tail-v2 payload from a generated WAV.

    The encoder appends a short tail to the audio. Each bit is one symbol
    window: one tone means 0 and a second tone means 1. The decoder looks only
    at the final payload-length tail and compares correlation against both
    tones for every bit.
    """
    audio = audio.astype(np.float32)
    symbol_seconds = 0.04
    segment_len = max(128, int(sample_rate * symbol_seconds))
    tag_len = segment_len * bits
    if len(audio) >= tag_len:
        audio = audio[-tag_len:]

    # These frequencies must match OutputProcessor.process().
    zero_freq = min(9000.0, sample_rate * 0.38)
    one_freq = min(10500.0, sample_rate * 0.44)
    bit_scores = []
    bit_confidence = []

    for bit_index in range(bits):
        start = bit_index * segment_len
        end = len(audio) if bit_index == bits - 1 else min(len(audio), start + segment_len)
        segment = audio[start:end]
        if len(segment) < 8:
            score = 0.0
            confidence = 0.0
            bit_scores.append(score)
            bit_confidence.append(confidence)
            continue

        window = np.hanning(len(segment)).astype(np.float32)
        t = np.arange(len(segment), dtype=np.float32) / float(sample_rate)

        # Compare how strongly the segment matches each possible bit tone.
        zero_ref = np.sin(2.0 * np.pi * zero_freq * t) * window
        one_ref = np.sin(2.0 * np.pi * one_freq * t) * window
        zero_energy = abs(float(np.dot(segment, zero_ref)))
        one_energy = abs(float(np.dot(segment, one_ref)))
        score = one_energy - zero_energy
        confidence = abs(score) / (one_energy + zero_energy + 1e-9)
        bit_scores.append(score)
        bit_confidence.append(float(min(1.0, confidence)))

    payload_id = 0
    for index, score in enumerate(bit_scores):
        if score > 0:
            payload_id += 1 << index

    return DecodeResult(
        payload_id=payload_id,
        bit_scores=bit_scores,
        bit_confidence=bit_confidence,
    )
