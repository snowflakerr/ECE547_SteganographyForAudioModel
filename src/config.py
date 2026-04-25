"""
config.py
=========
Centralised configuration dataclasses for the TraceableSpeech pipeline.
"""

from __future__ import annotations

import json
import dataclasses
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class AudioConfig:
    """Audio preprocessing parameters."""
    sample_rate: int   = 24_000
    n_mels:      int   = 80
    n_fft:       int   = 1_024
    hop_length:  int   = 256
    win_length:  int   = 1_024
    f_min:       float = 0.0
    f_max:       Optional[float] = None   # None → sample_rate / 2


@dataclass
class WatermarkConfig:
    """Payload encoding parameters.

    The watermark is *4 independent symbols*, each drawn from a 16-class
    vocabulary.  That gives 16^4 = 65 536 unique IDs.
    """
    num_symbols:   int = 4
    vocab_size:    int = 16
    embed_dim:     int = 16    # embedding dim per symbol
    watermark_dim: int = 256   # final projected conditioning vector


@dataclass
class ModelConfig:
    """Neural model architecture parameters."""
    resnet_feat_dim:    int   = 80
    resnet_embed_dim:   int   = 256
    resnet_num_heads:   int   = 8     # MQMHASTP attention heads
    resnet_num_queries: int   = 8     # MQMHASTP learnable query vectors
    injector_channels:  int   = 64
    injector_layers:    int   = 8
    injector_alpha:     float = 0.02  # default watermark strength scale


@dataclass
class TrainingConfig:
    batch_size:       int   = 16
    lr:               float = 1e-4
    num_epochs:       int   = 100
    warmup_steps:     int   = 1_000
    clip_grad_norm:   float = 5.0
    save_every:       int   = 10
    checkpoint_dir:   str   = "checkpoints"
    # Loss weights
    lambda_wm:        float = 1.0    # watermark cross-entropy
    lambda_recon:     float = 10.0   # waveform reconstruction (L1)
    lambda_mel:       float = 5.0    # mel reconstruction


@dataclass
class AugmentationConfig:
    """
    Defines the stochastic attack distribution used during training.
    Probabilities *must* sum to 1.0; a validation check is run at
    AugmentationPipeline construction time.

    Attack codes:
        CLP       — clean passthrough
        RSP-90    — resample 24 kHz → 21.6 kHz → 24 kHz
        Noise-W35 — additive white noise at 35 dB SNR
        APS-05    — amplitude scale × 0.5
        APS-15    — amplitude scale × 1.5
        HPF-1800  — high-pass filter at 1 800 Hz
        LPF-5000  — low-pass filter at 5 000 Hz
        MF-3      — sliding median filter (window = 3)
        TS-09     — time-stretch × 0.95 (speed change)
    """
    attack_schedule: List[Tuple[str, float]] = field(default_factory=lambda: [
        ("CLP",       0.20),
        ("RSP-90",    0.10),
        ("Noise-W35", 0.10),
        ("APS-05",    0.10),
        ("APS-15",    0.10),
        ("HPF-1800",  0.10),
        ("LPF-5000",  0.10),
        ("MF-3",      0.10),
        ("TS-09",     0.10),
    ])
    clip_prob: float = 0.5   # independent probability of temporal clipping


@dataclass
class Config:
    """Top-level config.  Serialisable to / from JSON."""

    audio:        AudioConfig        = field(default_factory=AudioConfig)
    watermark:    WatermarkConfig    = field(default_factory=WatermarkConfig)
    model:        ModelConfig        = field(default_factory=ModelConfig)
    training:     TrainingConfig     = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    # ── factory ──────────────────────────────────────────────────────────────

    @classmethod
    def default(cls) -> Config:
        return cls()

    # ── serialisation ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> Config:
        with open(path) as fh:
            d = json.load(fh)
        return cls(
            audio        = AudioConfig(**d["audio"]),
            watermark    = WatermarkConfig(**d["watermark"]),
            model        = ModelConfig(**d["model"]),
            training     = TrainingConfig(**d["training"]),
            augmentation = AugmentationConfig(**d["augmentation"]),
        )
