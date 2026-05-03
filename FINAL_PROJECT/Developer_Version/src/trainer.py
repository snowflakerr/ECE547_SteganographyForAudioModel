"""
trainer.py
==========
End-to-end watermark training loop.

Training objective
------------------
For each batch of clean audio waveforms:
  1. Sample a random watermark payload (4 symbols × 16 classes).
  2. Encode it to a conditioning vector.
  3. WaveformInjector generates an additive perturbation; waveform ← waveform + α·pert.
  4. Stochastic attack augmentation is applied to the watermarked audio.
  5. WatermarkDecoder recovers the payload from the attacked mel spectrogram.
  6. Total loss = λ_wm · sign_loss  +  λ_recon · L1(watermarked, clean)  +  λ_mel · L1(mel_wm, mel_clean)

"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import Config
from watermark_net import WatermarkSystem, random_watermark, sign_loss, accuracy
from audio_utils import MelExtractor
from augmentations import AugmentationPipeline

logger = logging.getLogger(__name__)


# ── training metrics ──────────────────────────────────────────────────────────

class RunningMetrics:
    """Accumulate scalar metrics and compute means on demand."""

    def __init__(self) -> None:
        self._data: Dict[str, list] = {}

    def update(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self._data.setdefault(k, []).append(float(v))

    def means(self) -> Dict[str, float]:
        return {k: sum(v) / len(v) for k, v in self._data.items() if v}

    def reset(self) -> None:
        self._data.clear()


# ── trainer ───────────────────────────────────────────────────────────────────

class WatermarkTrainer:
    """
    Trains a WatermarkSystem end-to-end.

    Parameters
    ----------
    system    : WatermarkSystem  — encoder + injector + decoder.
    mel       : MelExtractor
    augmenter : AugmentationPipeline
    cfg       : Config
    device    : str | torch.device
    """

    def __init__(
        self,
        system:    WatermarkSystem,
        mel:       MelExtractor,
        augmenter: AugmentationPipeline,
        cfg:       Config,
        device:    str | torch.device = "cpu",
    ) -> None:
        self.device    = torch.device(device)
        self.system    = system.to(self.device)
        self.mel       = mel.to(self.device)
        self.augmenter = augmenter
        self.cfg       = cfg
        self.tcfg      = cfg.training

        self.optimizer = AdamW(
            self.system.parameters(),
            lr=self.tcfg.lr,
            weight_decay=1e-4,
        )
        self.scheduler: Optional[CosineAnnealingLR] = None

        self._step      = 0
        self._epoch     = 0
        self._best_acc  = 0.0

        Path(self.tcfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # ── train ─────────────────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   Optional[DataLoader] = None,
    ) -> None:
        """Run the full training loop for ``cfg.training.num_epochs`` epochs."""
        total_steps = len(train_loader) * self.tcfg.num_epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=self.tcfg.lr * 0.01
        )

        for epoch in range(self._epoch, self.tcfg.num_epochs):
            self._epoch = epoch
            t0          = time.time()

            train_metrics = self.train_epoch(train_loader)
            elapsed       = time.time() - t0

            log_str = (
                f"Epoch {epoch+1:03d}/{self.tcfg.num_epochs}  "
                f"[{elapsed:.1f}s]  "
                + "  ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
            )

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                log_str += "  |  val: " + "  ".join(
                    f"{k}={v:.4f}" for k, v in val_metrics.items()
                )
                acc = val_metrics.get("acc", 0.0)
            else:
                acc = train_metrics.get("acc", 0.0)

            logger.info(log_str)

            if (epoch + 1) % self.tcfg.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1:03d}.pt")

            if acc > self._best_acc:
                self._best_acc = acc
                self.save_checkpoint("best.pt")
                logger.info("  ↑ New best accuracy: %.4f", acc)

        self.save_checkpoint("final.pt")

    # ── epoch ─────────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Train for one full pass over ``loader``."""
        self.system.train()
        metrics = RunningMetrics()

        for batch in loader:
            waveforms, _ = batch
            waveforms    = waveforms.to(self.device)    # [B, 1, T]
            B            = waveforms.size(0)

            # 1. Random watermark payload
            sign = random_watermark(
                B,
                self.cfg.watermark.num_symbols,
                self.cfg.watermark.vocab_size,
                device=self.device,
            )                                            # [B, 4]

            # 2–4. Embed → attack → decode
            losses, acc = self._forward_and_loss(waveforms, sign)
            total_loss  = losses["total"]

            # 5. Optimise
            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(
                self.system.parameters(), self.tcfg.clip_grad_norm
            )
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()

            self._step += 1
            metrics.update(
                loss      = total_loss.item(),
                loss_wm   = losses["wm"].item(),
                loss_recon= losses["recon"].item(),
                loss_mel  = losses["mel"].item(),
                acc       = acc,
            )

        return metrics.means()

    # ── validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on ``loader`` without gradient computation."""
        self.system.eval()
        metrics = RunningMetrics()

        for batch in loader:
            waveforms, _ = batch
            waveforms    = waveforms.to(self.device)
            B            = waveforms.size(0)

            sign = random_watermark(
                B,
                self.cfg.watermark.num_symbols,
                self.cfg.watermark.vocab_size,
                device=self.device,
            )

            losses, acc = self._forward_and_loss(waveforms, sign)
            metrics.update(
                loss = losses["total"].item(),
                acc  = acc,
            )

        return metrics.means()

    # ── core forward + loss ───────────────────────────────────────────────────

    def _forward_and_loss(
        self,
        waveforms: torch.Tensor,    # [B, 1, T]
        sign:      torch.Tensor,    # [B, num_symbols]
    ) -> tuple[Dict[str, torch.Tensor], float]:
        """Embed → attack → decode → compute losses."""
        tcfg = self.tcfg

        # Clean mel (for reconstruction loss)
        mel_clean = self.mel(waveforms)                    # [B, n_mels, T']

        # Embed
        out         = self.system.embed(waveforms, sign)
        watermarked = out.watermarked_waveform             # [B, 1, T]

        # Attack
        attacked, _ = self.augmenter(watermarked.detach())
        attacked    = attacked.to(self.device)

        # Decode
        mel_attacked = self.mel(attacked)                  # [B, n_mels, T'']
        mel_t        = mel_attacked.transpose(1, 2)        # [B, T'', n_mels]
        scores, pred = self.system.decode(mel_t)

        # Losses
        wm_loss    = sign_loss(scores, sign)
        recon_loss = F.l1_loss(watermarked, waveforms)
        mel_wm     = self.mel(watermarked)
        # Match lengths for mel loss (attacked mel may be shorter due to TS)
        min_t  = min(mel_clean.size(-1), mel_wm.size(-1))
        mel_loss = F.l1_loss(mel_wm[..., :min_t], mel_clean[..., :min_t])

        total = (
            tcfg.lambda_wm    * wm_loss
            + tcfg.lambda_recon * recon_loss
            + tcfg.lambda_mel   * mel_loss
        )

        acc = accuracy(pred, sign)

        return {
            "total": total,
            "wm":    wm_loss,
            "recon": recon_loss,
            "mel":   mel_loss,
        }, acc

    # ── checkpoint ────────────────────────────────────────────────────────────

    def save_checkpoint(self, filename: str) -> None:
        path = Path(self.tcfg.checkpoint_dir) / filename
        torch.save(
            {
                "epoch":      self._epoch,
                "step":       self._step,
                "best_acc":   self._best_acc,
                "model":      self.system.state_dict(),
                "optimizer":  self.optimizer.state_dict(),
                "scheduler":  self.scheduler.state_dict() if self.scheduler else None,
                "config":     self.cfg.to_dict(),
            },
            path,
        )
        logger.info("Checkpoint saved → %s", path)

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.system.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt["scheduler"] and self.scheduler:
            self.scheduler.load_state_dict(ckpt["scheduler"])
        self._epoch    = ckpt.get("epoch", 0)
        self._step     = ckpt.get("step", 0)
        self._best_acc = ckpt.get("best_acc", 0.0)
        logger.info(
            "Loaded checkpoint '%s' (epoch %d, best_acc=%.4f)",
            path, self._epoch, self._best_acc,
        )
