"""
detector.py
===========
Watermark detection and decoding from audio files or waveform tensors.

Provides per-symbol confidence scores so the caller can decide whether
a detected payload meets a threshold before trusting it.

Usage
-----
    from config import Config
    from watermark_net import WatermarkSystem
    from detector import WatermarkDetector

    cfg     = Config.default()
    system  = WatermarkSystem(cfg)
    # ... load checkpoint ...

    detector = WatermarkDetector(system, cfg)

    result = detector.detect_file("speech.wav")
    if result.is_confident(threshold=0.6):
        print("Detected payload:", result.sign)
    else:
        print("Low-confidence detection")

    detector.print_result(result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from config import Config
from audio_utils import MelExtractor, load_audio
from watermark_net import WatermarkSystem

logger = logging.getLogger(__name__)


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class DetectionResult:
    """Structured output of a watermark detection run.

    Attributes
    ----------
    sign              : recovered payload as a list of ints.
    symbol_probs      : list of ``num_symbols`` softmax probability vectors,
                        each of shape [vocab_size].
    symbol_confidence : max softmax probability per symbol (∈ [0, 1]).
    mean_confidence   : mean of symbol_confidence scores.
    source_path       : path that was decoded (if called via detect_file).
    """
    sign:              List[int]
    symbol_probs:      List[torch.Tensor]           # list of [vocab_size]
    symbol_confidence: List[float]
    mean_confidence:   float
    source_path:       Optional[str] = None

    def is_confident(self, threshold: float = 0.5) -> bool:
        """Return True if *all* symbols exceed ``threshold`` confidence."""
        return all(c >= threshold for c in self.symbol_confidence)

    def to_id(self) -> int:
        """Encode the 4-symbol payload as a single integer ID.

        ID = s[0] + s[1]*V + s[2]*V² + s[3]*V³  where V = vocab_size (16).
        """
        v   = len(self.symbol_probs[0])   # vocab_size
        return sum(s * (v ** i) for i, s in enumerate(self.sign))

    def __repr__(self) -> str:
        conf = [f"{c:.3f}" for c in self.symbol_confidence]
        return (
            f"DetectionResult("
            f"sign={self.sign}, "
            f"confidence={conf}, "
            f"mean={self.mean_confidence:.3f})"
        )


# ── detector ─────────────────────────────────────────────────────────────────

class WatermarkDetector:
    """
    Detects and decodes watermarks from waveforms or audio files.

    Parameters
    ----------
    system  : WatermarkSystem  — trained encoder + injector + decoder.
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

    # ── public API ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def detect(self, waveform: torch.Tensor) -> DetectionResult:
        """Decode the watermark from a waveform tensor.

        Parameters
        ----------
        waveform : [1, T] or [B, 1, T]  float32 audio (B=1 for single clip).

        Returns
        -------
        DetectionResult
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)    # [1, 1, T]
        waveform = waveform.to(self.device)

        mel     = self.mel(waveform)            # [1, n_mels, T']
        mel_t   = mel.transpose(1, 2)           # [1, T', n_mels]

        scores, pred_symbols = self.system.decode(mel_t)
        # scores : tuple of [1, vocab_size]

        sign       = pred_symbols.squeeze(0).tolist()
        sym_probs  = [F.softmax(s.squeeze(0), dim=0).cpu() for s in scores]
        sym_conf   = [float(p.max().item()) for p in sym_probs]
        mean_conf  = sum(sym_conf) / len(sym_conf)

        return DetectionResult(
            sign              = sign,
            symbol_probs      = sym_probs,
            symbol_confidence = sym_conf,
            mean_confidence   = mean_conf,
        )

    def detect_file(self, path: Union[str, Path]) -> DetectionResult:
        """Load an audio file and detect its watermark.

        Parameters
        ----------
        path : path to a .wav / .flac / etc. audio file.
        """
        path     = Path(path)
        waveform, _ = load_audio(
            path,
            target_sr = self.cfg.audio.sample_rate,
            mono      = True,
            normalise = True,
        )
        result = self.detect(waveform)
        result.source_path = str(path)
        return result

    def detect_batch(
        self, paths: List[Union[str, Path]]
    ) -> List[DetectionResult]:
        """Detect watermarks in a list of files.

        Returns one DetectionResult per file.
        """
        return [self.detect_file(p) for p in paths]

    # ── reporting ─────────────────────────────────────────────────────────────

    @staticmethod
    def print_result(result: DetectionResult, threshold: float = 0.5) -> None:
        """Pretty-print a detection result to stdout."""
        w = 50
        print("─" * w)
        print(" Watermark Detection Report")
        print("─" * w)

        if result.source_path:
            print(f" File      : {result.source_path}")

        print(f" Payload   : {result.sign}  →  ID {result.to_id()}")
        print(f" Mean conf : {result.mean_confidence:.3f}")
        print()
        print(" Per-symbol breakdown:")
        for i, (sym, conf, probs) in enumerate(
            zip(result.sign, result.symbol_confidence, result.symbol_probs)
        ):
            bar  = "█" * int(conf * 20)
            flag = "✓" if conf >= threshold else "✗"
            print(f"   s{i+1}={sym:2d}  conf={conf:.3f}  {flag}  {bar}")

        verdict = "DETECTED" if result.is_confident(threshold) else "UNCERTAIN"
        print()
        print(f" Verdict   : {verdict}  (threshold={threshold})")
        print("─" * w)

    # ── utility ───────────────────────────────────────────────────────────────

    def compare(
        self,
        result:         DetectionResult,
        expected_sign:  List[int],
    ) -> dict:
        """Compare a detected payload against a known ground-truth payload.

        Returns a dict with:
            ``correct``       — bool (all symbols match)
            ``symbol_match``  — per-symbol bool list
            ``accuracy``      — fraction of correct symbols
        """
        matches   = [p == e for p, e in zip(result.sign, expected_sign)]
        return {
            "correct":      all(matches),
            "symbol_match": matches,
            "accuracy":     sum(matches) / len(matches),
        }

    def __repr__(self) -> str:
        return (
            f"WatermarkDetector("
            f"device={self.device}, "
            f"n_mels={self.cfg.audio.n_mels})"
        )
