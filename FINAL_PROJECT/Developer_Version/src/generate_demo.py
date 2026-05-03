"""
generate_demo.py
================
Generates four demo audio files:

  hello_world_clean.wav          — plain TTS, no watermark
  hello_world_watermarked.wav    — TTS with inaudible spread-spectrum watermark
  scotty_5s_clean.wav            — first 5 s of Scotty Doesn't Know, no watermark
  scotty_5s_watermarked.wav      — first 5 s with the same watermark payload

Watermarking strategy
---------------------
The spread-spectrum watermark is embedded in the STFT magnitude domain using
the FramewiseWatermarker from watermark_core.py.  The original phase is kept
intact, which eliminates the phasiness artefacts that Griffin-Lim introduces.
Alpha is set to 0.01 (well below the psychoacoustic noise floor); the
perceptual RMS weighting further suppresses the watermark in quiet frames.
(AKA you shouldn't be able to hear)

Usage
-----
    python generate_demo.py
    python generate_demo.py --payload 42 --key "my_secret" --out_dir ./out
"""

from __future__ import annotations

import argparse
import hashlib
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import stft, istft

# ── inline spread-spectrum watermarker ───────────────────────────────────────
# (mirrors watermark_core.py so this script is self-contained)

class FramewiseWatermarker:
    """Spread-spectrum framewise watermarker operating on STFT magnitude."""

    MAX_PAYLOAD_BITS = 64

    def __init__(
        self,
        secret_key:   str   = "secret_key",
        alpha:        float = 0.01,
        payload_bits: int   = 32,
        perceptual:   bool  = True,
    ):
        if payload_bits > self.MAX_PAYLOAD_BITS:
            raise ValueError(f"payload_bits must be ≤ {self.MAX_PAYLOAD_BITS}")
        self.secret_key   = secret_key
        self.alpha        = alpha
        self.payload_bits = payload_bits
        self.perceptual   = perceptual
        self._seed        = self._key_to_seed(secret_key)

    @staticmethod
    def _key_to_seed(key: str) -> int:
        digest = hashlib.sha256(key.encode()).digest()
        return struct.unpack("<Q", digest[:8])[0]

    def _pn_matrix(self, n_frames: int, n_bins: int) -> np.ndarray:
        rng = np.random.default_rng(self._seed)
        return rng.choice(np.array([-1.0, 1.0]), size=(n_frames, n_bins)).astype(np.float32)

    def _bpsk_encode(self, payload: int) -> np.ndarray:
        bits = np.array([(payload >> i) & 1 for i in range(self.payload_bits)], dtype=np.float32)
        return 2.0 * bits - 1.0

    def embed(self, magnitude: np.ndarray, payload: int = 0) -> np.ndarray:
        """
        Embed watermark into an STFT magnitude spectrogram.

        Parameters
        ----------
        magnitude : [n_bins, n_frames]  — real-valued STFT magnitude
        payload   : int watermark ID

        Returns
        -------
        watermarked magnitude  [n_bins, n_frames]
        """
        magnitude = magnitude.astype(np.float32)
        n_bins, n_frames = magnitude.shape

        pn            = self._pn_matrix(n_frames, n_bins)           # [n_frames, n_bins]
        symbols       = self._bpsk_encode(payload)
        frame_symbols = symbols[np.arange(n_frames) % self.payload_bits]

        wm = (frame_symbols[:, None] * pn).T                        # [n_bins, n_frames]

        if self.perceptual:
            # Weight by local frame RMS so quiet frames carry no watermark
            frame_rms = np.sqrt(np.mean(magnitude ** 2, axis=0, keepdims=True) + 1e-8)
            wm = wm * frame_rms

        watermarked = magnitude + self.alpha * wm
        # Magnitudes must be non-negative
        return np.clip(watermarked, 0.0, None)


# ── audio helpers ─────────────────────────────────────────────────────────────

def synthesise_tts(text: str, sample_rate: int = 22050) -> np.ndarray:
    """Use espeak-ng to synthesise text → mono float32 waveform."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    result = subprocess.run(
        ["espeak-ng", "-w", tmp_path, "-a", "150", "-s", "150", text],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"espeak-ng failed: {result.stderr.decode()}")

    audio, sr = sf.read(tmp_path)
    Path(tmp_path).unlink()

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample to target sr if needed (espeak outputs 22050 by default)
    if sr != sample_rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sample_rate, sr)
        audio = resample_poly(audio, sample_rate // g, sr // g)

    return audio.astype(np.float32)


def load_and_trim(path: str, duration_s: float, sample_rate: int = 44100) -> np.ndarray:
    """Load a wav, mix to mono, trim to duration_s, return float32."""
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != sample_rate:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sample_rate, sr)
        audio = resample_poly(audio, sample_rate // g, sr // g)
        audio = audio.astype(np.float32)

    n = int(duration_s * sample_rate)
    return audio[:n].astype(np.float32)


def apply_watermark(
    audio: np.ndarray,
    sample_rate: int,
    watermarker: FramewiseWatermarker,
    payload: int,
) -> np.ndarray:
    """
    Embed watermark via STFT magnitude → reconstruct with original phase.

    Keeps original phase intact (no Griffin-Lim), so the only audible
    difference would come from the magnitude perturbation — which at
    alpha=0.01 with perceptual weighting is ~40–50 dB below the signal.
    """
    nperseg = 1024
    noverlap = 768     # 75 % overlap for smooth reconstruction

    # Forward STFT
    freqs, times, Zxx = stft(
        audio,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )

    magnitude = np.abs(Zxx)          # [n_bins, n_frames]
    phase     = np.angle(Zxx)        # [n_bins, n_frames]

    # Embed in magnitude domain
    wm_magnitude = watermarker.embed(magnitude, payload)

    # Reconstruct with original phase
    Zxx_wm = wm_magnitude * np.exp(1j * phase)

    _, audio_wm = istft(
        Zxx_wm,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=True,
    )

    # Trim/pad to original length
    audio_wm = audio_wm[: len(audio)]
    if len(audio_wm) < len(audio):
        audio_wm = np.pad(audio_wm, (0, len(audio) - len(audio_wm)))

    # Normalise to original peak so loudness is identical
    orig_peak = np.abs(audio).max()
    wm_peak   = np.abs(audio_wm).max()
    if wm_peak > 0:
        audio_wm = audio_wm * (orig_peak / wm_peak)

    return audio_wm.astype(np.float32)


def snr_db(original: np.ndarray, watermarked: np.ndarray) -> float:
    """Signal-to-watermark-noise ratio in dB."""
    noise = watermarked - original
    sig_p  = np.mean(original ** 2)
    noi_p  = np.mean(noise ** 2) + 1e-12
    return 10 * np.log10(sig_p / noi_p)


def save(path: str, audio: np.ndarray, sample_rate: int) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate, subtype="PCM_16")
    dur = len(audio) / sample_rate
    print(f"  ✓  {path}  ({dur:.2f}s)")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate watermark demo audio files")
    parser.add_argument("--payload",  type=int,   default=42,
                        help="Integer watermark payload (default: 42)")
    parser.add_argument("--key",      type=str,   default="traceable_speech_demo",
                        help="Shared secret for PN carrier")
    parser.add_argument("--alpha",    type=float, default=0.01,
                        help="Watermark embedding strength (default: 0.01, inaudible)")
    parser.add_argument("--scotty",   type=str,
                        default=str(Path(__file__).with_name("scotty.wav")),
                        help="Path to scotty.wav")
    parser.add_argument("--out_dir",  type=str,
                        default=str(Path(__file__).with_name("out")),
                        help="Output directory")
    args = parser.parse_args()

    out = args.out_dir
    wm  = FramewiseWatermarker(
        secret_key   = args.key,
        alpha        = args.alpha,
        payload_bits = 32,
        perceptual   = True,
    )

    print(f"\nWatermark config:")
    print(f"  payload  = {args.payload}")
    print(f"  key      = '{args.key}'")
    print(f"  alpha    = {args.alpha}  (perceptual RMS-weighted)")
    print()

    # ── 1 & 2 : Hello World TTS ──────────────────────────────────────────────
    print("Synthesising TTS…")
    TTS_SR = 22050
    hello  = synthesise_tts("Hello, world.", sample_rate=TTS_SR)
    peak   = np.abs(hello).max()
    if peak > 0:
        hello = hello / peak * 0.9    # normalise to -0.9 dBFS

    print("  Generating hello_world pair…")
    save(f"{out}/hello_world_clean.wav",       hello,                          TTS_SR)
    hello_wm = apply_watermark(hello, TTS_SR, wm, args.payload)
    save(f"{out}/hello_world_watermarked.wav", hello_wm,                       TTS_SR)
    print(f"  SNR (hello): {snr_db(hello, hello_wm):.1f} dB")

    # ── 3 & 4 : Scotty Doesn't Know — first 5 s ──────────────────────────────
    print(f"\nLoading '{args.scotty}'…")
    SCOTTY_SR = 44100
    scotty    = load_and_trim(args.scotty, duration_s=5.0, sample_rate=SCOTTY_SR)
    peak      = np.abs(scotty).max()
    if peak > 0:
        scotty = scotty / peak * 0.9

    print("  Generating scotty_5s pair…")
    save(f"{out}/scotty_5s_clean.wav",       scotty,                           SCOTTY_SR)
    scotty_wm = apply_watermark(scotty, SCOTTY_SR, wm, args.payload)
    save(f"{out}/scotty_5s_watermarked.wav", scotty_wm,                        SCOTTY_SR)
    print(f"  SNR (scotty): {snr_db(scotty, scotty_wm):.1f} dB")

    print("\nDone.  All four files written to", out)


if __name__ == "__main__":
    main()
