"""
tts_pipe.py
===========
Text-to-speech pipeline with post-hoc neural watermark injection.

The pipeline:
  1. Synthesises speech via Coqui TTS (or any backend returning a waveform).
  2. Runs the waveform through WaveformInjector, conditioned on a watermark ID.
  3. Saves the watermarked audio and returns it alongside metadata.

Usage
-----
    from config import Config
    from watermark_net import WatermarkSystem
    from audio_utils import MelExtractor
    from tts_pipe import WatermarkedTTSPipeline

    cfg    = Config.default()
    system = WatermarkSystem(cfg)
    # ... load checkpoint into system ...

    pipe = WatermarkedTTSPipeline(system, cfg)
    result = pipe.synthesise(
        text    = "Hello, world.",
        sign    = [3, 7, 1, 12],          # explicit watermark ID (optional)
        out_path= "hello.wav",
    )
    print(result.sign, result.duration_s)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import torch
import numpy as np

from config import Config
from audio_utils import MelExtractor, save_audio
from watermark_net import WatermarkSystem, random_watermark

logger = logging.getLogger(__name__)

# Optional Coqui TTS import — graceful error if not installed
try:
    from TTS.api import TTS as CoquiTTS
    _COQUI_AVAILABLE = True
except ImportError:
    _COQUI_AVAILABLE = False
    logger.warning(
        "Coqui TTS not installed.  Install with:  pip install TTS\n"
        "WatermarkedTTSPipeline will raise RuntimeError if synthesis is attempted."
    )


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class TTSSynthesisResult:
    waveform:   torch.Tensor    # [1, T]  watermarked audio
    sign:       List[int]       # watermark payload as a list of ints
    sample_rate: int
    out_path:   Optional[str]
    duration_s: float

    def __repr__(self) -> str:
        return (
            f"TTSSynthesisResult("
            f"sign={self.sign}, "
            f"duration={self.duration_s:.2f}s, "
            f"out='{self.out_path}')"
        )


# ── pipeline ──────────────────────────────────────────────────────────────────

class WatermarkedTTSPipeline:
    """
    Text → watermarked speech pipeline.

    Parameters
    ----------
    system     : WatermarkSystem  — trained encoder + injector.
    cfg        : Config
    model_name : str              — Coqui TTS model string.
    device     : str | torch.device
    """

    DEFAULT_MODEL = "tts_models/en/ljspeech/vits"

    def __init__(
        self,
        system:     WatermarkSystem,
        cfg:        Config,
        model_name: str                      = DEFAULT_MODEL,
        device:     Union[str, torch.device] = "cpu",
    ) -> None:
        self.system = system.to(device)
        self.system.eval()
        self.cfg        = cfg
        self.device     = torch.device(device)
        self.mel        = MelExtractor(cfg.audio).to(device)

        if not _COQUI_AVAILABLE:
            self._tts = None
        else:
            logger.info("Loading Coqui TTS model: %s", model_name)
            self._tts = CoquiTTS(model_name=model_name)
            logger.info("TTS model ready.")

    # ── public API ────────────────────────────────────────────────────────────

    def synthesise(
        self,
        text:     str,
        sign:     Optional[List[int]] = None,
        out_path: Optional[str]       = None,
    ) -> TTSSynthesisResult:
        """Synthesise speech and embed a watermark.

        Parameters
        ----------
        text     : text to synthesise.
        sign     : watermark payload as a list of ``num_symbols`` ints, each
                   in [0, vocab_size).  If None, a random payload is used.
        out_path : if provided, save the result to this path.

        Returns
        -------
        TTSSynthesisResult
        """
        if self._tts is None:
            raise RuntimeError(
                "Coqui TTS is not installed.  Run: pip install TTS"
            )

        sr     = self.cfg.audio.sample_rate
        wm_cfg = self.cfg.watermark

        # 1. Validate / generate payload
        sign_tensor = self._prepare_sign(sign, wm_cfg.num_symbols, wm_cfg.vocab_size)

        # 2. TTS synthesis → numpy array
        logger.debug("Synthesising: '%s'", text)
        wav_np = self._tts.tts(text=text)                    # list or np.array
        wav_np = np.asarray(wav_np, dtype=np.float32)

        # 3. Convert to tensor [1, 1, T]
        waveform = torch.from_numpy(wav_np).unsqueeze(0).unsqueeze(0).to(self.device)

        # 4. Embed watermark
        watermarked = self._embed(waveform, sign_tensor)     # [1, 1, T]

        out_wav = watermarked.squeeze(0)                     # [1, T]
        dur     = out_wav.size(-1) / sr

        # 5. Optionally save
        if out_path:
            save_audio(out_wav, out_path, sr)
            logger.info("Saved watermarked speech → %s  (%.2f s)", out_path, dur)

        return TTSSynthesisResult(
            waveform    = out_wav.cpu(),
            sign        = sign_tensor.squeeze(0).tolist(),
            sample_rate = sr,
            out_path    = out_path,
            duration_s  = dur,
        )

    # ── inject-only mode (no TTS) ─────────────────────────────────────────────

    def embed_waveform(
        self,
        waveform: torch.Tensor,
        sign:     Optional[List[int]] = None,
        out_path: Optional[str]       = None,
    ) -> TTSSynthesisResult:
        """Embed a watermark into an *existing* waveform (no TTS step).

        Useful when the audio comes from a non-Coqui source.

        Parameters
        ----------
        waveform : [1, T] or [1, 1, T]  float32 audio.
        sign     : watermark payload; None → random.
        out_path : optional save path.
        """
        sr     = self.cfg.audio.sample_rate
        wm_cfg = self.cfg.watermark

        sign_tensor = self._prepare_sign(sign, wm_cfg.num_symbols, wm_cfg.vocab_size)

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)   # → [1, 1, T]
        waveform = waveform.to(self.device)

        watermarked = self._embed(waveform, sign_tensor)
        out_wav     = watermarked.squeeze(0).cpu()
        dur         = out_wav.size(-1) / sr

        if out_path:
            save_audio(out_wav, out_path, sr)

        return TTSSynthesisResult(
            waveform    = out_wav,
            sign        = sign_tensor.squeeze(0).tolist(),
            sample_rate = sr,
            out_path    = out_path,
            duration_s  = dur,
        )

    # ── internals ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _embed(
        self,
        waveform:    torch.Tensor,    # [B, 1, T]
        sign_tensor: torch.Tensor,    # [B, num_symbols]
    ) -> torch.Tensor:
        """Run encoder + injector and return watermarked waveform."""
        out = self.system.embed(waveform, sign_tensor)
        return out.watermarked_waveform

    def _prepare_sign(
        self,
        sign:        Optional[List[int]],
        num_symbols: int,
        vocab_size:  int,
    ) -> torch.Tensor:
        """Validate and convert sign to [1, num_symbols] long tensor."""
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

        t = torch.tensor([sign], dtype=torch.long, device=self.device)
        return t

    def __repr__(self) -> str:
        return (
            f"WatermarkedTTSPipeline("
            f"device={self.device}, "
            f"alpha={self.system.alpha})"
        )
