"""
augmentations.py
================
Stochastic augmentation / attack pipeline used during watermark training.

All attacks operate on waveform tensors of shape [B, C, T]  (float32, ±1).

Usage
-----
    cfg     = AugmentationConfig()
    pipeline = AugmentationPipeline(cfg, sample_rate=24_000)
    attacked, op_name = pipeline(waveform)

    attacked = pipeline.apply("Noise-W35", waveform)
"""

from __future__ import annotations

import random
import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

from config import AugmentationConfig

logger = logging.getLogger(__name__)


class AugmentationPipeline:
    """
    Stochastic attack pipeline for watermark robustness training.

    Parameters
    ----------
    cfg         : AugmentationConfig  — schedule and clip probability.
    sample_rate : int                 — native sample rate of the audio.
    """

    def __init__(
        self,
        cfg:         AugmentationConfig,
        sample_rate: int = 24_000,
    ) -> None:
        self.cfg         = cfg
        self.sample_rate = sample_rate
        self._validate_schedule()

        # Build (op_name, cumulative_prob) list for fast sampling
        self._ops: list[str]   = []
        self._probs: list[float] = []
        cum = 0.0
        for name, p in cfg.attack_schedule:
            cum += p
            self._ops.append(name)
            self._probs.append(cum)

    # ── validation ────────────────────────────────────────────────────────────

    def _validate_schedule(self) -> None:
        total = sum(p for _, p in self.cfg.attack_schedule)
        if abs(total - 1.0) > 1e-4:
            raise ValueError(
                f"AugmentationConfig.attack_schedule probabilities sum to "
                f"{total:.4f} — must sum to 1.0."
            )

    # ── sampling ──────────────────────────────────────────────────────────────

    def sample_op(self) -> str:
        """Draw one operation name according to the configured distribution."""
        r = random.random()
        for op, cum in zip(self._ops, self._probs):
            if r <= cum:
                return op
        # Fallback: should only be reached due to floating-point rounding
        return self._ops[-1]

    # ── dispatch ──────────────────────────────────────────────────────────────

    def __call__(
        self, waveform: torch.Tensor
    ) -> Tuple[torch.Tensor, str]:
        """Apply a randomly sampled attack then optionally clip.

        Parameters
        ----------
        waveform : [B, C, T]

        Returns
        -------
        (attacked_waveform, op_name)
        """
        op       = self.sample_op()
        attacked = self.apply(op, waveform)

        # Independent temporal-clip augmentation
        if random.random() < self.cfg.clip_prob:
            attacked, _ = self.temporal_clip(attacked)

        return attacked, op

    def apply(self, op: str, waveform: torch.Tensor) -> torch.Tensor:
        """Apply a named attack to a waveform.

        Parameters
        ----------
        op       : str          — attack code (e.g. "Noise-W35").
        waveform : [B, C, T]

        Returns
        -------
        [B, C, T]  (may be shorter in time if TS-09 is applied)
        """
        dispatch = {
            "CLP":       self._clp,
            "RSP-90":    self._rsp_90,
            "Noise-W35": self._noise_w35,
            "APS-05":    self._aps_05,
            "APS-15":    self._aps_15,
            "HPF-1800":  self._hpf_1800,
            "LPF-5000":  self._lpf_5000,
            "MF-3":      self._mf_3,
            "TS-09":     self._ts_09,
        }
        fn = dispatch.get(op)
        if fn is None:
            logger.warning("Unknown attack op '%s' — falling back to CLP.", op)
            return waveform
        return fn(waveform)

    # ── attack implementations ────────────────────────────────────────────────

    @staticmethod
    def _clp(x: torch.Tensor) -> torch.Tensor:
        """Clean passthrough."""
        return x

    def _rsp_90(self, x: torch.Tensor) -> torch.Tensor:
        """Resample 24 kHz → 21.6 kHz → 24 kHz (90 % speed perturbation)."""
        sr      = self.sample_rate
        sr_down = int(sr * 0.9)
        down = torchaudio.transforms.Resample(sr, sr_down).to(x.device)
        up   = torchaudio.transforms.Resample(sr_down, sr).to(x.device)
        return up(down(x))

    @staticmethod
    def _noise_w35(x: torch.Tensor) -> torch.Tensor:
        """Add white noise at 35 dB SNR."""
        snr       = 10 ** (35.0 / 10.0)
        power     = x.pow(2).mean()
        noise_std = torch.sqrt(power / snr + 1e-12)
        noise     = torch.randn_like(x) * noise_std
        return x + noise

    @staticmethod
    def _aps_05(x: torch.Tensor) -> torch.Tensor:
        """Amplitude scale × 0.5."""
        return x * 0.5

    @staticmethod
    def _aps_15(x: torch.Tensor) -> torch.Tensor:
        """Amplitude scale × 1.5 (clip to ±1)."""
        return (x * 1.5).clamp(-1.0, 1.0)

    def _hpf_1800(self, x: torch.Tensor) -> torch.Tensor:
        """High-pass biquad filter at 1 800 Hz."""
        return torchaudio.functional.highpass_biquad(
            x, self.sample_rate, cutoff_freq=1800.0, Q=0.707
        )

    def _lpf_5000(self, x: torch.Tensor) -> torch.Tensor:
        """Low-pass biquad filter at 5 000 Hz."""
        return torchaudio.functional.lowpass_biquad(
            x, self.sample_rate, cutoff_freq=5000.0, Q=0.707
        )

    @staticmethod
    def _mf_3(x: torch.Tensor) -> torch.Tensor:
        """Sliding median filter with window = 3 along the time axis.

        FIX: original code called ``torch.median(window)`` which returns a
        scalar over the entire tensor.  Correct approach uses ``unfold`` to
        extract per-position windows and ``median(-1)`` for per-sample medians.
        """
        window_size = 3
        pad         = window_size // 2
        x_padded    = F.pad(x, (pad, pad), mode="reflect")
        # unfold last dim → [B, C, T, window_size]
        windows     = x_padded.unfold(-1, window_size, 1)
        return windows.median(dim=-1).values

    def _ts_09(self, x: torch.Tensor) -> torch.Tensor:
        """Time-stretch × 0.95 via resampling (slightly slower)."""
        sr      = self.sample_rate
        sr_new  = int(sr * 0.95)
        resamp  = torchaudio.transforms.Resample(sr, sr_new).to(x.device)
        return resamp(x)

    # ── temporal clipping ─────────────────────────────────────────────────────

    @staticmethod
    def temporal_clip(
        x: torch.Tensor,
        min_cut_frac: float = 0.10,
        max_cut_frac: float = 0.25,
        num_cuts:     int   = 2,
    ) -> Tuple[torch.Tensor, bool]:
        """Remove up to ``num_cuts`` random contiguous segments from the audio.

        FIX: original code computed ``y[:, :, :cut_end - cut_length]`` where
        both values were randomly drawn, meaning the slice could be negative.
        This version draws a *start index* so the cut is always valid.

        Parameters
        ----------
        x            : [B, C, T]
        min_cut_frac : minimum cut length as fraction of T.
        max_cut_frac : maximum cut length as fraction of T.
        num_cuts     : number of independent cuts.

        Returns
        -------
        (clipped_tensor, was_clipped_bool)
        """
        if random.random() > 0.5:
            return x, False

        T = x.size(2)
        for _ in range(num_cuts):
            T_cur = x.size(2)
            if T_cur < 8:
                break
            cut_len   = random.randint(
                max(1, int(T_cur * min_cut_frac)),
                max(2, int(T_cur * max_cut_frac)),
            )
            cut_start = random.randint(0, T_cur - cut_len)
            x = torch.cat(
                [x[:, :, :cut_start], x[:, :, cut_start + cut_len:]],
                dim=2,
            )
        return x, True

    # ── convenience ───────────────────────────────────────────────────────────

    def list_ops(self) -> list[str]:
        """Return the list of registered operation names."""
        return list(self._ops)

    def __repr__(self) -> str:
        schedule = ", ".join(
            f"{op}:{p:.2f}"
            for op, p in self.cfg.attack_schedule
        )
        return (
            f"AugmentationPipeline(sr={self.sample_rate}, "
            f"clip_prob={self.cfg.clip_prob}, schedule=[{schedule}])"
        )
