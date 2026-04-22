"""
audio_utils.py
==============
Audio I/O and mel-spectrogram extraction utilities.

Classes
-------
MelExtractor  — waveform [B, 1, T] → mel [B, n_mels, T']
AudioDataset  — minimal torch Dataset wrapping a directory of .wav files

Functions
---------
load_audio    — path → (waveform [1, T], sr)  with optional resampling
save_audio    — waveform [1, T] + path → writes .wav
pad_or_trim   — pad / crop a waveform to a fixed number of samples
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset

from config import AudioConfig

logger = logging.getLogger(__name__)


# ── mel spectrogram extractor ─────────────────────────────────────────────────

class MelExtractor(nn.Module):
    """Differentiable mel-spectrogram extractor.

    Parameters
    ----------
    cfg : AudioConfig
    """

    def __init__(self, cfg: AudioConfig) -> None:
        super().__init__()
        self.cfg    = cfg
        f_max       = cfg.f_max or cfg.sample_rate // 2

        self.mel_transform = T.MelSpectrogram(
            sample_rate    = cfg.sample_rate,
            n_fft          = cfg.n_fft,
            win_length     = cfg.win_length,
            hop_length     = cfg.hop_length,
            n_mels         = cfg.n_mels,
            f_min          = cfg.f_min,
            f_max          = f_max,
            power          = 2.0,
            normalized     = False,
            center         = True,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : [B, 1, T]  or  [1, T]  (auto-unsqueezed)

        Returns
        -------
        mel : [B, n_mels, T']  log-mel spectrogram (dB-scaled)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        if waveform.dim() != 3:
            raise ValueError(f"Expected 2-D or 3-D waveform, got {waveform.dim()}-D")

        # Move transform to the same device as input
        self.mel_transform = self.mel_transform.to(waveform.device)
        self.amplitude_to_db = self.amplitude_to_db.to(waveform.device)

        # Merge batch and channel dimensions for torchaudio
        B, C, T_in = waveform.shape
        wav_2d = waveform.reshape(B * C, T_in)

        mel_power = self.mel_transform(wav_2d)               # [B*C, n_mels, T']
        mel_db    = self.amplitude_to_db(mel_power)          # dB scale

        _, M, T_out = mel_db.shape
        return mel_db.reshape(B, C, M, T_out).squeeze(1)     # [B, n_mels, T']

    def extra_repr(self) -> str:
        return (
            f"sr={self.cfg.sample_rate}, n_mels={self.cfg.n_mels}, "
            f"n_fft={self.cfg.n_fft}, hop={self.cfg.hop_length}"
        )


# ── audio I/O ─────────────────────────────────────────────────────────────────

def load_audio(
    path:            str | Path,
    target_sr:       Optional[int]  = None,
    mono:            bool           = True,
    normalise:       bool           = True,
) -> Tuple[torch.Tensor, int]:
    """Load an audio file and optionally resample / normalise.

    Parameters
    ----------
    path       : path to audio file (any format torchaudio supports).
    target_sr  : if provided, resample to this sample rate.
    mono       : if True, average channels to mono.
    normalise  : if True, peak-normalise to ±1.

    Returns
    -------
    (waveform, sample_rate)
    waveform : [1, T]  float32
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))    # [C, T]

    if mono and waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if target_sr is not None and target_sr != sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform  = resampler(waveform)
        sr        = target_sr

    if normalise:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    return waveform, sr


def save_audio(
    waveform:    torch.Tensor,
    path:        str | Path,
    sample_rate: int,
) -> None:
    """Save a waveform tensor to a .wav file.

    Parameters
    ----------
    waveform    : [1, T] or [C, T]  float32
    path        : output path (parent directory created if needed).
    sample_rate : audio sample rate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = waveform.detach().cpu().float()
    torchaudio.save(str(path), wav, sample_rate)
    logger.debug("Saved %s  (%.2f s)", path, wav.size(-1) / sample_rate)


def pad_or_trim(
    waveform:   torch.Tensor,
    num_samples: int,
    mode:       str = "zero",
) -> torch.Tensor:
    """Pad or crop a waveform to exactly ``num_samples`` samples.

    Parameters
    ----------
    waveform    : [..., T]
    num_samples : desired number of samples.
    mode        : "zero" (zero-pad) or "repeat" (tile the signal).

    Returns
    -------
    [..., num_samples]
    """
    T = waveform.size(-1)
    if T == num_samples:
        return waveform
    if T > num_samples:
        return waveform[..., :num_samples]

    # Pad
    deficit = num_samples - T
    if mode == "zero":
        pad_shape = list(waveform.shape)
        pad_shape[-1] = deficit
        padding = torch.zeros(pad_shape, dtype=waveform.dtype, device=waveform.device)
        return torch.cat([waveform, padding], dim=-1)
    elif mode == "repeat":
        repeats = math.ceil(num_samples / T)
        tiled   = waveform.repeat(*([1] * (waveform.dim() - 1)), repeats)
        return tiled[..., :num_samples]
    else:
        raise ValueError(f"Unknown pad mode: '{mode}'")


# ── dataset ───────────────────────────────────────────────────────────────────

class AudioDataset(Dataset):
    """Minimal dataset that returns fixed-length mono waveforms from a directory.

    Recursively finds .wav / .flac / .mp3 files under ``root``.

    Parameters
    ----------
    root          : directory to search.
    sample_rate   : target sample rate (resamples on load if needed).
    segment_secs  : clip length in seconds (None → return full file).
    max_files     : cap on dataset size (None → all files).
    """

    EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".opus"}

    def __init__(
        self,
        root:         str | Path,
        sample_rate:  int            = 24_000,
        segment_secs: Optional[float] = 2.0,
        max_files:    Optional[int]   = None,
    ) -> None:
        self.root         = Path(root)
        self.sample_rate  = sample_rate
        self.segment_len  = (
            int(segment_secs * sample_rate) if segment_secs else None
        )

        self.files = sorted([
            p for p in self.root.rglob("*")
            if p.suffix.lower() in self.EXTENSIONS
        ])

        if not self.files:
            raise FileNotFoundError(f"No audio files found under {self.root}")

        if max_files is not None:
            self.files = self.files[:max_files]

        logger.info(
            "AudioDataset: %d files under %s  (sr=%d, seg=%.1fs)",
            len(self.files), self.root, sample_rate,
            segment_secs if segment_secs else -1,
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns
        -------
        (waveform [1, T], file_path_str)
        """
        path              = self.files[idx]
        waveform, _       = load_audio(path, target_sr=self.sample_rate)

        if self.segment_len is not None:
            waveform = self._random_crop_or_pad(waveform, self.segment_len)

        return waveform, str(path)

    @staticmethod
    def _random_crop_or_pad(wav: torch.Tensor, length: int) -> torch.Tensor:
        T = wav.size(-1)
        if T >= length:
            start = torch.randint(0, T - length + 1, (1,)).item()
            return wav[:, start: start + length]
        return pad_or_trim(wav, length)
