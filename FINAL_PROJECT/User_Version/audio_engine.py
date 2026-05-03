from __future__ import annotations

import hashlib
import math
import struct
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


SAMPLE_RATE = 24_000


class OutputProcessor:
    """Final audio post-processor used by both generator apps.

    The generated speech is left intact, then a short high-frequency payload
    tag is appended to the end of the waveform. Each bit is represented by one
    of two tones. This is intentionally simple and robust for class demos: the
    checker can recover the payload without needing the original clean audio.
    """

    MAX_BITS = 64
    SYMBOL_SECONDS = 0.04

    def __init__(self, secret_key: str, strength: float = 0.03, bits: int = 32) -> None:
        """Configure payload encoding.

        secret_key is kept for interface symmetry with the checker and future
        keyed encoders. strength controls the quiet tag amplitude relative to
        the generated speech.
        """
        if bits > self.MAX_BITS:
            raise ValueError(f"bits must be <= {self.MAX_BITS}")
        self.secret_key = secret_key
        self.strength = strength
        self.bits = bits
        self._seed = self._key_to_seed(secret_key)

    @staticmethod
    def _key_to_seed(key: str) -> int:
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return struct.unpack("<Q", digest[:8])[0]

    def _carrier(self, frames: int, bins: int) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        return rng.choice(np.array([-1.0, 1.0]), size=(frames, bins)).astype(np.float32)

    def _bit_carrier(self, bit_index: int, frames: int, bins: int) -> np.ndarray:
        seed = self._seed ^ ((bit_index + 1) * 0x9E3779B97F4A7C15)
        rng = np.random.default_rng(seed & ((1 << 64) - 1))
        return rng.choice(np.array([-1.0, 1.0]), size=(bins, frames)).astype(np.float32)

    def _symbols(self, output_id: int) -> np.ndarray:
        """Convert an integer payload ID into BPSK-style bit symbols."""
        values = np.array([(output_id >> i) & 1 for i in range(self.bits)], dtype=np.float32)
        return 2.0 * values - 1.0

    def process(self, audio: np.ndarray, output_id: int, sample_rate: int) -> np.ndarray:
        """Append the encoded payload tag and return the final waveform."""
        final_audio = audio.astype(np.float32)
        symbols = self._symbols(output_id)

        # Scale the tag to the source audio so quiet clips do not get a
        # disproportionately loud tag, while still keeping a practical floor.
        rms = float(np.sqrt(np.mean(final_audio**2) + 1e-12))
        amplitude = max(0.002, self.strength * 0.5 * rms)

        # Use high frequencies so the tag is easy for the checker to isolate
        # and less intrusive than tones in the main speech band.
        zero_freq = min(9000.0, sample_rate * 0.38)
        one_freq = min(10500.0, sample_rate * 0.44)
        segment_len = max(128, int(sample_rate * self.SYMBOL_SECONDS))
        tag = np.zeros(segment_len * self.bits, dtype=np.float32)

        for bit_index, symbol in enumerate(symbols):
            start = bit_index * segment_len
            end = start + segment_len
            t = np.arange(end - start, dtype=np.float32) / float(sample_rate)
            freq = one_freq if symbol > 0 else zero_freq
            envelope = np.hanning(len(t)).astype(np.float32) if len(t) > 3 else np.ones_like(t)
            tag[start:end] = amplitude * np.sin(2.0 * np.pi * freq * t) * envelope

        final_audio = np.concatenate([final_audio, tag])
        return np.clip(final_audio, -1.0, 1.0).astype(np.float32)


def peak_normalize(audio: np.ndarray, target: float = 0.9) -> np.ndarray:
    """Normalize generated speech to a consistent peak level before tagging."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * target
    return audio.astype(np.float32)


def resample(audio: np.ndarray, source_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Resample Coqui output to the project-wide sample rate."""
    if source_sr == target_sr:
        return audio.astype(np.float32)
    from math import gcd

    factor = gcd(source_sr, target_sr)
    return resample_poly(audio, target_sr // factor, source_sr // factor).astype(np.float32)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Save a WAV and return bytes for Streamlit playback/download widgets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path.read_bytes()
