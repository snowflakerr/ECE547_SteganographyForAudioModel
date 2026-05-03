from __future__ import annotations

import hashlib
import struct
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import istft, resample_poly, stft


SAMPLE_RATE = 24_000
ENCODER_VERSION = "stft-spread-spectrum-v1"


class OutputProcessor:
    """STFT magnitude-domain spread-spectrum audio watermarker.

    This replaces the older FSK-tail demo tag. The watermark is embedded inside
    the generated audio's STFT magnitude frames, while the original phase is
    preserved during reconstruction. The output stays the same length as the
    input instead of appending a removable tag to the end of the WAV.
    """

    MAX_BITS = 64

    def __init__(self, secret_key: str, strength: float = 0.01, bits: int = 32) -> None:
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

    def _pn_matrix(self, n_frames: int, n_bins: int) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        return rng.choice(np.array([-1.0, 1.0]), size=(n_frames, n_bins)).astype(np.float32)

    def _symbols(self, output_id: int) -> np.ndarray:
        bits = np.array([(output_id >> i) & 1 for i in range(self.bits)], dtype=np.float32)
        return 2.0 * bits - 1.0

    def process(self, audio: np.ndarray, output_id: int, sample_rate: int) -> np.ndarray:
        """Embed payload into STFT magnitude and reconstruct same-length audio."""
        clean = audio.astype(np.float32)
        original_len = len(clean)
        if original_len == 0:
            return clean

        nperseg = 1024
        noverlap = 768

        _, _, zxx = stft(
            clean,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
            padded=True,
        )

        magnitude = np.abs(zxx).astype(np.float32)
        phase = np.angle(zxx).astype(np.float32)
        n_bins, n_frames = magnitude.shape

        pn = self._pn_matrix(n_frames, n_bins).T  # [n_bins, n_frames]
        symbols = self._symbols(output_id)
        frame_symbols = symbols[np.arange(n_frames) % self.bits][None, :]

        # Perceptual-ish RMS weighting: quieter frames receive less watermark
        # energy, making the perturbation less obvious during silence.
        frame_rms = np.sqrt(np.mean(magnitude**2, axis=0, keepdims=True) + 1e-8)
        watermark = pn * frame_symbols * frame_rms

        watermarked_magnitude = np.clip(magnitude + self.strength * watermark, 0.0, None)
        zxx_watermarked = watermarked_magnitude * np.exp(1j * phase)

        _, reconstructed = istft(
            zxx_watermarked,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary=True,
        )

        reconstructed = reconstructed[:original_len]
        if len(reconstructed) < original_len:
            reconstructed = np.pad(reconstructed, (0, original_len - len(reconstructed)))

        # Match original peak level so watermarking does not change loudness.
        clean_peak = float(np.max(np.abs(clean)))
        wm_peak = float(np.max(np.abs(reconstructed)))
        if clean_peak > 0 and wm_peak > 0:
            reconstructed = reconstructed * (clean_peak / wm_peak)

        return np.clip(reconstructed, -1.0, 1.0).astype(np.float32)


def peak_normalize(audio: np.ndarray, target: float = 0.9) -> np.ndarray:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * target
    return audio.astype(np.float32)


def resample(audio: np.ndarray, source_sr: int, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32)
    factor = gcd(source_sr, target_sr)
    return resample_poly(audio, target_sr // factor, source_sr // factor).astype(np.float32)


def save_wav(path: Path, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    return path.read_bytes()
