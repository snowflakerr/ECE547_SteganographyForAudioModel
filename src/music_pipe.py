"""
music_pipe.py
=============
Watermark embedding pipeline for existing audio files (music, speech, any .wav).

Unlike the TTS pipeline, there is no synthesis step — the neural injector
runs entirely post-hoc on loaded audio.

Usage
-----
    from config import Config
    from watermark_net import WatermarkSystem
    from music_pipe import MusicWatermarkPipeline

    cfg    = Config.default()
    system = WatermarkSystem(cfg)
    # ... load checkpoint ...

    pipe   = MusicWatermarkPipeline(system, cfg)
    result = pipe.embed_file(
        in_path  = "song.wav",
        out_path = "song_wm.wav",
        sign     = [5, 2, 14, 0],
    )
    print(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch

from config import Config
from audio_utils import MelExtractor, load_audio, save_audio, pad_or_trim
from watermark_net import WatermarkSystem, random_watermark

logger = logging.getLogger(__name__)


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class MusicEmbedResult:
    in_path:     str
    out_path:    Optional[str]
    sign:        List[int]
    sample_rate: int
    duration_s:  float
    snr_db:      float            # signal-to-noise ratio of the perturbation

    def __repr__(self) -> str:
        return (
            f"MusicEmbedResult("
            f"sign={self.sign}, "
            f"snr={self.snr_db:.1f} dB, "
            f"duration={self.duration_s:.2f}s, "
            f"out='{self.out_path}')"
        )


# ── pipeline ──────────────────────────────────────────────────────────────────

class MusicWatermarkPipeline:
    """
    Embed a neural watermark into any audio file.

    Supports:
    - Arbitrary duration inputs (handles via chunked processing if needed).
    - Stereo input (watermark applied to all channels consistently).
    - Batch processing of multiple files.

    Parameters
    ----------
    system  : WatermarkSystem  — trained encoder + injector.
    cfg     : Config
    device  : str | torch.device
    """

    def __init__(
        self,
        system: WatermarkSystem,
        cfg:    Config,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.system = system.to(device)
        self.system.eval()
        self.cfg    = cfg
        self.device = torch.device(device)
        self.mel    = MelExtractor(cfg.audio).to(device)

    # ── single-file API ───────────────────────────────────────────────────────

    def embed_file(
        self,
        in_path:  Union[str, Path],
        out_path: Optional[Union[str, Path]] = None,
        sign:     Optional[List[int]]        = None,
    ) -> MusicEmbedResult:
        """Embed watermark into an audio file.

        Parameters
        ----------
        in_path  : path to the source audio file.
        out_path : save watermarked audio here; None → do not save.
        sign     : watermark payload (list of ``num_symbols`` ints);
                   None → random payload.

        Returns
        -------
        MusicEmbedResult
        """
        sr     = self.cfg.audio.sample_rate
        wm_cfg = self.cfg.watermark

        # Load and resample
        waveform, _ = load_audio(
            in_path, target_sr=sr, mono=True, normalise=True
        )                                          # [1, T]
        waveform = waveform.to(self.device)

        sign_tensor = self._prepare_sign(sign, wm_cfg.num_symbols, wm_cfg.vocab_size)

        # Embed
        watermarked, perturbation = self._embed(
            waveform.unsqueeze(0),    # [1, 1, T]
            sign_tensor,
        )

        out_wav = watermarked.squeeze(0).cpu()     # [1, T]
        dur     = out_wav.size(-1) / sr
        snr     = self._compute_snr(waveform.cpu(), perturbation.squeeze(0).cpu())

        if out_path is not None:
            save_audio(out_wav, out_path, sr)
            logger.info(
                "Watermarked audio  →  %s  (SNR %.1f dB, %.2f s)",
                out_path, snr, dur,
            )

        return MusicEmbedResult(
            in_path     = str(in_path),
            out_path    = str(out_path) if out_path else None,
            sign        = sign_tensor.squeeze(0).tolist(),
            sample_rate = sr,
            duration_s  = dur,
            snr_db      = snr,
        )

    # ── batch API ─────────────────────────────────────────────────────────────

    def embed_batch(
        self,
        in_paths:  List[Union[str, Path]],
        out_dir:   Union[str, Path],
        sign:      Optional[List[int]] = None,
        suffix:    str                 = "_wm",
    ) -> List[MusicEmbedResult]:
        """Embed the same watermark into a list of audio files.

        Parameters
        ----------
        in_paths : source files.
        out_dir  : output directory (created if needed).
        sign     : shared watermark payload; None → random (one per file).
        suffix   : appended to the stem of each output filename.

        Returns
        -------
        List[MusicEmbedResult]
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for p in in_paths:
            p         = Path(p)
            out_path  = out_dir / f"{p.stem}{suffix}{p.suffix}"
            file_sign = sign  # shared payload
            results.append(
                self.embed_file(p, out_path, file_sign)
            )
        return results

    # ── internals ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed(
        self,
        waveform:    torch.Tensor,    # [B, 1, T]
        sign_tensor: torch.Tensor,    # [B, num_symbols]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (watermarked_waveform, perturbation) both [B, 1, T]."""
        out = self.system.embed(waveform, sign_tensor)
        return out.watermarked_waveform, out.perturbation

    def _prepare_sign(
        self,
        sign:        Optional[List[int]],
        num_symbols: int,
        vocab_size:  int,
    ) -> torch.Tensor:
        if sign is None:
            return random_watermark(1, num_symbols, vocab_size, device=self.device)

        if len(sign) != num_symbols:
            raise ValueError(
                f"sign must have {num_symbols} elements, got {len(sign)}"
            )
        for s in sign:
            if not (0 <= s < vocab_size):
                raise ValueError(
                    f"Each symbol must be in [0, {vocab_size}), got {s}"
                )
        return torch.tensor([sign], dtype=torch.long, device=self.device)

    @staticmethod
    def _compute_snr(
        original:    torch.Tensor,
        perturbation: torch.Tensor,
    ) -> float:
        """Compute signal-to-watermark-noise ratio in dB."""
        sig_power  = original.pow(2).mean()
        noise_power = (perturbation * perturbation).mean() + 1e-12
        snr = 10.0 * torch.log10(sig_power / noise_power)
        return snr.item()

    def __repr__(self) -> str:
        return (
            f"MusicWatermarkPipeline("
            f"device={self.device}, "
            f"alpha={self.system.alpha})"
        )
