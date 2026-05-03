"""
watermark_net.py
================
All neural watermarking components.

Classes
-------
WatermarkEncoder   — integer payload → conditioning vector
FiLM               — Feature-wise Linear Modulation conditioning layer
WaveformInjector   — generates an additive watermark perturbation
WatermarkDecoder   — mel spectrogram → decoded payload
WatermarkSystem    — convenience wrapper that owns all three components

Helpers
-------
random_watermark   — sample a random batch of payloads
sign_loss          — multi-symbol cross-entropy loss
accuracy           — per-batch symbol accuracy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import ResNet34
from config import Config, WatermarkConfig, ModelConfig


# ── payload helpers ───────────────────────────────────────────────────────────

def random_watermark(batch_size: int, num_symbols: int = 4, vocab_size: int = 16,
                     device: torch.device | str = "cpu") -> torch.Tensor:
    """Sample a random watermark payload tensor.

    Returns
    -------
    sign : [B, num_symbols]  long tensor, values in [0, vocab_size).
    """
    return torch.randint(
        low=0, high=vocab_size,
        size=(batch_size, num_symbols),
        device=device,
    )


def sign_loss(
    score_tuple: Tuple[torch.Tensor, ...],
    target: torch.Tensor,
) -> torch.Tensor:
    """Averaged cross-entropy over all num_symbols classification heads.

    Parameters
    ----------
    score_tuple : tuple of [B, vocab_size] logit tensors, len = num_symbols
    target      : [B, num_symbols]  long tensor

    Returns
    -------
    scalar loss
    """
    targets = [t.squeeze(1) for t in target.split(1, dim=1)]
    loss = sum(F.cross_entropy(s, t) for s, t in zip(score_tuple, targets))
    return loss / len(score_tuple)


def accuracy(
    pred_symbols: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Mean per-symbol accuracy over the batch.

    Parameters
    ----------
    pred_symbols : [B, num_symbols]  long tensor (argmax output)
    target       : [B, num_symbols]  long tensor
    """
    correct = (pred_symbols == target).float()
    return correct.mean().item()


# ── watermark encoder ─────────────────────────────────────────────────────────

class WatermarkEncoder(nn.Module):
    """Encodes a multi-symbol integer payload into a conditioning vector.

    Architecture: per-symbol embedding → flatten → two-layer MLP → L2-norm.

    Parameters
    ----------
    cfg : WatermarkConfig
    """

    def __init__(self, cfg: WatermarkConfig) -> None:
        super().__init__()
        self.cfg = cfg
        flat_dim = cfg.num_symbols * cfg.embed_dim

        self.embedding    = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.linear1      = nn.Linear(flat_dim, cfg.watermark_dim // 2)
        self.linear2      = nn.Linear(cfg.watermark_dim // 2, cfg.watermark_dim)
        self.norm         = nn.LayerNorm(cfg.watermark_dim)

        nn.init.normal_(self.embedding.weight, 0, 0.01)

    def forward(self, sign: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        sign : [B, num_symbols]  long tensor

        Returns
        -------
        vec : [B, watermark_dim]
        """
        emb  = self.embedding(sign)                          # [B, S, embed_dim]
        flat = emb.reshape(emb.size(0), -1)                  # [B, S*embed_dim]
        h    = F.leaky_relu(self.linear1(flat), 0.1)
        h    = self.linear2(h)
        return self.norm(h)                                  # [B, watermark_dim]


# ── FiLM conditioning layer ───────────────────────────────────────────────────

class FiLM(nn.Module):
    """Feature-wise Linear Modulation.

    Applies a learned affine transform to a feature map, conditioned on
    an external vector:  out = γ(cond) ⊙ x  +  β(cond).
    """

    def __init__(self, cond_dim: int, feat_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feat_dim)
        self.beta  = nn.Linear(cond_dim, feat_dim)
        # Initialise as near-identity
        nn.init.ones_(self.gamma.weight.data[torch.arange(min(cond_dim, feat_dim)),
                                              torch.arange(min(cond_dim, feat_dim))]
                       if cond_dim == feat_dim else self.gamma.weight.data)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : [B, C, T]
        cond : [B, cond_dim]
        """
        g = self.gamma(cond).unsqueeze(-1)    # [B, C, 1]
        b = self.beta(cond).unsqueeze(-1)     # [B, C, 1]
        return g * x + b


# ── waveform injector ─────────────────────────────────────────────────────────

class WaveformInjector(nn.Module):
    """Generates an additive watermark perturbation at the waveform level.

    The network takes the clean audio as a structural reference (allowing
    perceptual masking) and a watermark conditioning vector.  Dilated
    causal convolutions give a wide receptive field with few parameters.

    The output is passed through ``tanh`` so it is bounded in (−1, +1).
    The caller scales it by ``alpha`` before adding to the clean signal.

    Parameters
    ----------
    watermark_dim : int — dimensionality of the conditioning vector.
    channels      : int — internal channel width.
    num_layers    : int — number of dilated residual layers (dilation = 2^i).
    """

    def __init__(
        self,
        watermark_dim: int = 256,
        channels:      int = 64,
        num_layers:    int = 8,
    ) -> None:
        super().__init__()
        self.input_proj  = nn.Conv1d(1, channels, kernel_size=1)
        self.res_convs   = nn.ModuleList()
        self.films       = nn.ModuleList()
        self.skip_convs  = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            self.res_convs.append(
                nn.Conv1d(channels, channels, kernel_size=3,
                          padding=dilation, dilation=dilation)
            )
            self.films.append(FiLM(watermark_dim, channels))
            self.skip_convs.append(nn.Conv1d(channels, channels, kernel_size=1))

        self.output_proj = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        wm_vec:   torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        waveform : [B, 1, T]  clean audio in [−1, 1]
        wm_vec   : [B, watermark_dim]

        Returns
        -------
        perturbation : [B, 1, T]  additive watermark signal ∈ (−1, 1)
        """
        x = self.input_proj(waveform)    # [B, C, T]
        skip_sum = torch.zeros_like(x)

        for res_conv, film, skip_conv in zip(
            self.res_convs, self.films, self.skip_convs
        ):
            residual = x
            x = F.relu(res_conv(x), inplace=True)
            x = film(x, wm_vec)
            skip_sum = skip_sum + skip_conv(x)
            x = x + residual                    # residual connection

        return self.output_proj(skip_sum)        # [B, 1, T]


# ── watermark decoder ─────────────────────────────────────────────────────────

class WatermarkDecoder(nn.Module):
    """Recovers the watermark payload from a (possibly attacked) mel spectrogram.

    Architecture: ResNet-34 + MQMHASTP → one 2-layer MLP head per symbol.

    Parameters
    ----------
    wm_cfg    : WatermarkConfig
    model_cfg : ModelConfig
    """

    def __init__(self, wm_cfg: WatermarkConfig, model_cfg: ModelConfig) -> None:
        super().__init__()
        self.wm_cfg = wm_cfg

        self.backbone = ResNet34(
            feat_dim    = model_cfg.resnet_feat_dim,
            embed_dim   = model_cfg.resnet_embed_dim,
            num_heads   = model_cfg.resnet_num_heads,
            num_queries = model_cfg.resnet_num_queries,
        )

        # One 2-layer classification head per symbol
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(model_cfg.resnet_embed_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, wm_cfg.vocab_size),
            )
            for _ in range(wm_cfg.num_symbols)
        ])

    def forward(
        self, mel: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """
        Parameters
        ----------
        mel : [B, T, n_mels]  (time-first mel spectrogram)

        Returns
        -------
        score_tuple  : tuple of [B, vocab_size] logits, len = num_symbols
        pred_symbols : [B, num_symbols]  argmax prediction
        """
        _, embedding = self.backbone(mel)          # [B, embed_dim]

        scores = tuple(head(embedding) for head in self.heads)

        pred_symbols = torch.stack(
            [s.argmax(dim=1) for s in scores], dim=1
        )                                          # [B, num_symbols]

        return scores, pred_symbols


# ── system wrapper ────────────────────────────────────────────────────────────

@dataclass
class WatermarkSystemOutput:
    watermarked_waveform: torch.Tensor          # [B, 1, T]
    perturbation:         torch.Tensor          # [B, 1, T]
    wm_vector:            torch.Tensor          # [B, watermark_dim]


class WatermarkSystem(nn.Module):
    """Convenience wrapper owning encoder + injector + decoder.

    Useful for inference and for passing a single ``nn.Module`` to an
    optimiser.

    Parameters
    ----------
    cfg : Config
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg      = cfg
        self.encoder  = WatermarkEncoder(cfg.watermark)
        self.injector = WaveformInjector(
            watermark_dim = cfg.watermark.watermark_dim,
            channels      = cfg.model.injector_channels,
            num_layers    = cfg.model.injector_layers,
        )
        self.decoder  = WatermarkDecoder(cfg.watermark, cfg.model)
        self.alpha    = cfg.model.injector_alpha

    # ── embed ─────────────────────────────────────────────────────────────────

    def embed(
        self,
        waveform: torch.Tensor,
        sign:     torch.Tensor,
    ) -> WatermarkSystemOutput:
        """Embed watermark into a waveform batch.

        Parameters
        ----------
        waveform : [B, 1, T]
        sign     : [B, num_symbols]  long

        Returns
        -------
        WatermarkSystemOutput
        """
        wm_vec       = self.encoder(sign)
        perturbation = self.injector(waveform, wm_vec)
        watermarked  = (waveform + self.alpha * perturbation).clamp(-1.0, 1.0)
        return WatermarkSystemOutput(watermarked, perturbation, wm_vec)

    # ── decode ────────────────────────────────────────────────────────────────

    def decode(
        self, mel: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor]:
        """Decode watermark from mel spectrogram (time-first [B, T, F])."""
        return self.decoder(mel)

    # ── combined forward (training) ───────────────────────────────────────────

    def forward(
        self,
        waveform: torch.Tensor,
        sign:     torch.Tensor,
        mel_fn,                     # callable: waveform → mel [B, T, F]
        attack_fn=None,             # optional augmentation callable
    ) -> dict:
        """Full training forward pass.

        Returns a dict with all intermediate tensors needed for loss computation.
        """
        # 1. Embed
        out = self.embed(waveform, sign)

        # 2. Optionally attack
        attacked = attack_fn(out.watermarked_waveform) if attack_fn else out.watermarked_waveform

        # 3. Mel extraction + decode
        mel          = mel_fn(attacked)                     # [B, n_mels, T']
        mel_t        = mel.transpose(1, 2)                  # [B, T', n_mels]
        scores, pred = self.decode(mel_t)

        return {
            "watermarked":   out.watermarked_waveform,
            "perturbation":  out.perturbation,
            "wm_vector":     out.wm_vector,
            "attacked":      attacked,
            "scores":        scores,
            "pred_symbols":  pred,
        }
