from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Developer_Version"))
sys.path.insert(0, str(ROOT / "Watermark_Checker"))

from audio_engine import OutputProcessor  # noqa: E402
from watermark_codec import decode_payload  # noqa: E402


def test_blind_decode_recovers_low_strength_stft_payload() -> None:
    sample_rate = 24_000
    seconds = 5.0
    rng = np.random.default_rng(547)
    noise = rng.normal(0.0, 0.02, int(sample_rate * seconds)).astype(np.float32)
    t = np.arange(noise.size, dtype=np.float32) / sample_rate
    clean = (
        0.32 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.14 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.05 * np.sin(2.0 * np.pi * 880.0 * t)
        + noise
    ).astype(np.float32)

    payload = 5
    key = "voice_studio_private_key"
    watermarked = OutputProcessor(key, strength=0.03, bits=32).process(clean, payload, sample_rate)

    result = decode_payload(watermarked, sample_rate, key, bits=32)

    assert result.payload_id == payload
