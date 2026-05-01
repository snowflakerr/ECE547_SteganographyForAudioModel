"""
resnet.py
=========
ResNet-34 backbone for the watermark decoder, ending with Multi-Query
Multi-Head Attentive Statistics Temporal Pooling (MQMHASTP).

Architecture summary
--------------------
Input  : [B, T, feat_dim]   (mel spectrogram, time-first)
Reshape: [B, 1, feat_dim, T]
ResNet : 4 × block groups, stride only in the freq dimension → [B, 512, freq', T]
FreqPool: adaptive avg pool freq → 1 → [B, 512, T]
Project : linear 512 → embed_dim per timestep
MQMHASTP: learnable queries attend over T, output mean + std → [B, embed_dim]

Returns a 2-tuple  (frame_features, embedding)  so callers can use
    x = model(mel)
    embedding = x[-1]
which mirrors the original TraceableSpeech API.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Basic 2-D residual block ──────────────────────────────────────────────────

class BasicBlock2D(nn.Module):
    """Standard pre-activation ResNet basic block for 2-D feature maps."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: tuple[int, int] = (1, 1),
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False,
        )
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False,
        )
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != (1, 1) or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


# ── MQMHASTP pooling ──────────────────────────────────────────────────────────

class MQMHASTPPooling(nn.Module):
    """
    Multi-Query Multi-Head Attentive Statistics Temporal Pooling.

    For each of `num_queries` learnable query vectors, the module computes
    attention weights over the time axis T, then forms a weighted mean
    and weighted standard-deviation of the value sequence.  The resulting
    (mean ‖ std) vectors are concatenated across queries and projected to
    `embed_dim`.

    Parameters
    ----------
    embed_dim  : int — dimensionality of the input (and output) sequence.
    num_heads  : int — number of attention heads (embed_dim must be divisible).
    num_queries: int — number of learnable global query vectors.
    dropout    : float — attention dropout during training.
    """

    def __init__(
        self,
        embed_dim:   int   = 256,
        num_heads:   int   = 8,
        num_queries: int   = 8,
        dropout:     float = 0.1,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embed_dim   = embed_dim
        self.num_heads   = num_heads
        self.num_queries = num_queries
        self.scale       = (embed_dim // num_heads) ** -0.5

        # Learnable query bank: num_queries × embed_dim
        self.queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)

        self.key_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.val_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout  = nn.Dropout(dropout)

        # 2 × embed_dim per query (mean + std), projected back to embed_dim
        self.out_proj = nn.Linear(num_queries * 2 * embed_dim, embed_dim)
        self.norm     = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, T, embed_dim]

        Returns
        -------
        pooled : [B, embed_dim]
        """
        B, T, _ = x.shape

        K = self.key_proj(x)                               # [B, T, D]
        V = self.val_proj(x)                               # [B, T, D]
        Q = self.queries.unsqueeze(0).expand(B, -1, -1)    # [B, Q, D]

        # Scaled dot-product attention  [B, Q, T]
        attn = torch.bmm(Q, K.transpose(1, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted mean  [B, Q, D]
        mean    = torch.bmm(attn, V)
        # Weighted second moment → std
        mean_sq = torch.bmm(attn, V ** 2)
        std     = torch.sqrt(torch.clamp(mean_sq - mean ** 2, min=1e-8))

        # Concatenate statistics across queries  [B, Q*2*D]
        stats = torch.cat([mean, std], dim=-1)         # [B, Q, 2D]
        stats = stats.reshape(B, -1)                   # [B, Q*2D]

        return self.norm(self.out_proj(stats))          # [B, D]


# ── ResNet-34 ─────────────────────────────────────────────────────────────────

class ResNet34(nn.Module):
    """
    ResNet-34 for mel-spectrogram-based watermark recovery.

    Strides are applied only in the frequency (height) dimension so that the
    full temporal resolution is preserved for the MQMHASTP pooling layer.

    Parameters
    ----------
    feat_dim    : int — number of mel bins (height of the spectrogram).
    embed_dim   : int — output embedding dimensionality.
    num_heads   : int — MQMHASTP attention heads.
    num_queries : int — MQMHASTP query vectors.
    """

    _LAYER_CFG = [
        # (out_channels, num_blocks, freq_stride)
        (64,  3, 1),
        (128, 4, 2),
        (256, 6, 2),
        (512, 3, 2),
    ]

    def __init__(
        self,
        feat_dim:    int = 80,
        embed_dim:   int = 256,
        num_heads:   int = 8,
        num_queries: int = 8,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual stages
        in_ch = 64
        stages: list[nn.Module] = []
        for out_ch, n_blocks, freq_stride in self._LAYER_CFG:
            stage = self._make_stage(in_ch, out_ch, n_blocks, freq_stride)
            stages.append(stage)
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)

        # Pool frequency axis to 1 regardless of input height
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))

        # Project channel dim → embed_dim per time-step
        self.channel_proj = nn.Conv1d(512, embed_dim, kernel_size=1, bias=False)
        self.proj_norm    = nn.BatchNorm1d(embed_dim)

        # Temporal pooling
        self.pooling = MQMHASTPPooling(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_queries=num_queries,
        )

        self._init_weights()

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_stage(
        in_ch: int,
        out_ch: int,
        n_blocks: int,
        freq_stride: int,
    ) -> nn.Sequential:
        blocks: list[nn.Module] = [
            BasicBlock2D(in_ch, out_ch, stride=(freq_stride, 1))
        ]
        for _ in range(1, n_blocks):
            blocks.append(BasicBlock2D(out_ch, out_ch))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : [B, T, feat_dim]   (mel spectrogram, time-first)

        Returns
        -------
        (frame_features, embedding) — 2-tuple so ``result[-1]`` gives the
        final embedding, matching the original TraceableSpeech API.

        frame_features : [B, T', embed_dim]
        embedding      : [B, embed_dim]
        """
        # Reshape to [B, 1, feat_dim, T] for 2-D convolutions
        B, T, n_freq = x.shape
        h = x.transpose(1, 2).unsqueeze(1)    # [B, 1, n_freq, T]

        h = self.stem(h)                       # [B, 64, F, T]
        h = self.stages(h)                     # [B, 512, F', T]

        h = self.freq_pool(h)                  # [B, 512, 1, T]
        h = h.squeeze(2)                       # [B, 512, T]

        h = F.relu(self.proj_norm(self.channel_proj(h)), inplace=True)
        # h : [B, embed_dim, T]

        frame_features = h.transpose(1, 2)     # [B, T, embed_dim]
        embedding      = self.pooling(frame_features)  # [B, embed_dim]

        return frame_features, embedding
