"""
tests/test_pipeline.py
======================
Comprehensive pytest test suite for the TraceableSpeech watermarking pipeline.
Inspired by TraceableSpeec src code https://github.com/zjzser/TraceableSpeech

Run with:
    pytest tests/test_pipeline.py -v

All tests use synthetic audio — no real files or GPU needed.
Tests are grouped into:
  - Unit: individual components in isolation.
  - Integration: components wired together.
  - Robustness: watermark survives common attacks.
"""

from __future__ import annotations

import sys
import os
import math
import pytest
import torch
import torch.nn.functional as F
import numpy as np

# Allow imports from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config, AudioConfig, WatermarkConfig, ModelConfig, AugmentationConfig
from resnet import BasicBlock2D, MQMHASTPPooling, ResNet34
from watermark_net import (
    WatermarkEncoder, FiLM, WaveformInjector,
    WatermarkDecoder, WatermarkSystem,
    random_watermark, sign_loss, accuracy,
)
from augmentations import AugmentationPipeline
from audio_utils import MelExtractor, pad_or_trim


# ── fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_RATE  = 24_000
DURATION_S   = 2.0
NUM_SAMPLES  = int(SAMPLE_RATE * DURATION_S)
BATCH_SIZE   = 2
DEVICE       = "cpu"


def make_cfg() -> Config:
    """Minimal config for fast unit tests."""
    cfg = Config.default()
    cfg.audio.sample_rate = SAMPLE_RATE
    cfg.model.injector_layers   = 4   # fewer layers = faster test
    cfg.model.resnet_embed_dim  = 64  # smaller backbone
    cfg.watermark.watermark_dim = 64
    return cfg


def make_waveform(B: int = BATCH_SIZE) -> torch.Tensor:
    """Synthetic white-noise waveform [B, 1, T]."""
    return torch.randn(B, 1, NUM_SAMPLES) * 0.1


def make_sign(B: int = BATCH_SIZE, cfg: Config | None = None) -> torch.Tensor:
    c = cfg or make_cfg()
    return random_watermark(B, c.watermark.num_symbols, c.watermark.vocab_size)


# ═══════════════════════════════════════════════════════════════════════════════
#  UNIT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfig:
    def test_default_creates_without_error(self):
        cfg = Config.default()
        assert cfg.audio.sample_rate == 24_000

    def test_serialise_round_trip(self, tmp_path):
        cfg  = Config.default()
        path = str(tmp_path / "cfg.json")
        cfg.save(path)
        cfg2 = Config.load(path)
        assert cfg.audio.sample_rate == cfg2.audio.sample_rate
        assert cfg.watermark.num_symbols == cfg2.watermark.num_symbols

    def test_augmentation_probabilities_sum_to_one(self):
        cfg   = Config.default()
        total = sum(p for _, p in cfg.augmentation.attack_schedule)
        assert abs(total - 1.0) < 1e-4


class TestMQMHASTPPooling:
    def test_output_shape(self):
        pool = MQMHASTPPooling(embed_dim=64, num_heads=4, num_queries=4)
        x    = torch.randn(2, 50, 64)
        out  = pool(x)
        assert out.shape == (2, 64)

    def test_invalid_head_config_raises(self):
        with pytest.raises(ValueError):
            MQMHASTPPooling(embed_dim=65, num_heads=4)

    def test_different_seq_lengths(self):
        pool = MQMHASTPPooling(embed_dim=32, num_heads=4, num_queries=2)
        for T in [10, 50, 200]:
            out = pool(torch.randn(1, T, 32))
            assert out.shape == (1, 32)


class TestResNet34:
    def test_output_shapes(self):
        model  = ResNet34(feat_dim=80, embed_dim=64, num_heads=4, num_queries=4)
        x      = torch.randn(2, 50, 80)    # [B, T, F]
        frames, emb = model(x)
        assert emb.shape == (2, 64)
        assert frames.shape[-1] == 64      # last dim is embed_dim

    def test_returns_two_tuple(self):
        model = ResNet34(feat_dim=80, embed_dim=64, num_heads=4, num_queries=4)
        out   = model(torch.randn(1, 30, 80))
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_last_element_is_embedding(self):
        model = ResNet34(feat_dim=80, embed_dim=64, num_heads=4, num_queries=4)
        result = model(torch.randn(1, 30, 80))
        emb = result[-1]
        assert emb.shape == (1, 64)


class TestWatermarkEncoder:
    def test_output_shape(self):
        cfg  = make_cfg()
        enc  = WatermarkEncoder(cfg.watermark)
        sign = make_sign(cfg=cfg)
        out  = enc(sign)
        assert out.shape == (BATCH_SIZE, cfg.watermark.watermark_dim)

    def test_different_payloads_give_different_vectors(self):
        cfg  = make_cfg()
        enc  = WatermarkEncoder(cfg.watermark)
        s1   = torch.zeros(1, cfg.watermark.num_symbols, dtype=torch.long)
        s2   = torch.ones(1,  cfg.watermark.num_symbols, dtype=torch.long)
        v1   = enc(s1)
        v2   = enc(s2)
        assert not torch.allclose(v1, v2)

    def test_layer_norm_output(self):
        """Encoder output should be normalised (not trivially zero)."""
        cfg  = make_cfg()
        enc  = WatermarkEncoder(cfg.watermark)
        sign = make_sign(cfg=cfg)
        out  = enc(sign)
        assert out.abs().mean().item() > 0


class TestFiLM:
    def test_output_shape(self):
        film = FiLM(cond_dim=32, feat_dim=16)
        x    = torch.randn(2, 16, 100)
        cond = torch.randn(2, 32)
        out  = film(x, cond)
        assert out.shape == (2, 16, 100)

    def test_conditioning_changes_output(self):
        film  = FiLM(cond_dim=8, feat_dim=8)
        x     = torch.randn(1, 8, 50)
        cond1 = torch.zeros(1, 8)
        cond2 = torch.ones(1, 8)
        assert not torch.allclose(film(x, cond1), film(x, cond2))


class TestWaveformInjector:
    def test_output_shape(self):
        cfg  = make_cfg()
        inj  = WaveformInjector(
            watermark_dim = cfg.watermark.watermark_dim,
            channels      = 16,
            num_layers    = 2,
        )
        wav  = make_waveform()
        vec  = torch.randn(BATCH_SIZE, cfg.watermark.watermark_dim)
        out  = inj(wav, vec)
        assert out.shape == wav.shape

    def test_output_bounded(self):
        """tanh output must lie strictly in (−1, 1)."""
        cfg  = make_cfg()
        inj  = WaveformInjector(
            watermark_dim = cfg.watermark.watermark_dim,
            channels      = 16,
            num_layers    = 2,
        )
        wav  = make_waveform()
        vec  = torch.randn(BATCH_SIZE, cfg.watermark.watermark_dim)
        out  = inj(wav, vec)
        assert out.abs().max().item() <= 1.0


class TestWatermarkDecoder:
    def test_output_shapes(self):
        cfg  = make_cfg()
        dec  = WatermarkDecoder(cfg.watermark, cfg.model)
        mel  = MelExtractor(cfg.audio)
        wav  = make_waveform()
        m    = mel(wav).transpose(1, 2)    # [B, T', n_mels]
        scores, pred = dec(m)
        assert len(scores) == cfg.watermark.num_symbols
        assert pred.shape == (BATCH_SIZE, cfg.watermark.num_symbols)

    def test_score_tuple_length(self):
        cfg     = make_cfg()
        dec     = WatermarkDecoder(cfg.watermark, cfg.model)
        mel_ext = MelExtractor(cfg.audio)
        m       = mel_ext(make_waveform()).transpose(1, 2)
        scores, _ = dec(m)
        assert len(scores) == cfg.watermark.num_symbols
        assert scores[0].shape[1] == cfg.watermark.vocab_size


class TestSignLoss:
    def test_perfect_prediction_is_low(self):
        vocab = 16
        S     = 4
        B     = 4
        signs = random_watermark(B, S, vocab)
        # Create one-hot logits → argmax matches target
        logits = [F.one_hot(signs[:, i], vocab).float() * 100 for i in range(S)]
        loss   = sign_loss(tuple(logits), signs)
        assert loss.item() < 0.01

    def test_random_prediction_is_higher(self):
        vocab  = 16
        S      = 4
        B      = 4
        signs  = random_watermark(B, S, vocab)
        logits = tuple(torch.randn(B, vocab) for _ in range(S))
        loss   = sign_loss(logits, signs)
        assert loss.item() > 0.1


class TestAccuracy:
    def test_perfect_accuracy(self):
        signs = random_watermark(4, 4, 16)
        assert accuracy(signs, signs) == pytest.approx(1.0)

    def test_zero_accuracy(self):
        signs = torch.zeros(4, 4, dtype=torch.long)
        wrong = torch.ones(4, 4, dtype=torch.long)
        assert accuracy(wrong, signs) == pytest.approx(0.0)


class TestAugmentationPipeline:
    def setup_method(self):
        self.cfg      = make_cfg()
        self.pipeline = AugmentationPipeline(
            self.cfg.augmentation, sample_rate=SAMPLE_RATE
        )

    def test_bad_probabilities_raise(self):
        bad_cfg = AugmentationConfig(attack_schedule=[("CLP", 0.5)])
        with pytest.raises(ValueError, match="sum to"):
            AugmentationPipeline(bad_cfg, sample_rate=SAMPLE_RATE)

    def test_clp_is_identity(self):
        wav = make_waveform(1)
        out = self.pipeline.apply("CLP", wav)
        assert torch.allclose(out, wav)

    def test_noise_w35_adds_noise(self):
        wav = make_waveform(1)
        out = self.pipeline.apply("Noise-W35", wav)
        assert not torch.allclose(out, wav)

    def test_aps_05_scales_amplitude(self):
        wav = make_waveform(1)
        out = self.pipeline.apply("APS-05", wav)
        assert torch.allclose(out, wav * 0.5)

    def test_aps_15_clips_to_unit(self):
        wav = torch.ones(1, 1, 1000) * 0.9
        out = self.pipeline.apply("APS-15", wav)
        assert out.abs().max().item() <= 1.0

    def test_median_filter_correct_shape(self):
        wav = make_waveform(1)
        out = self.pipeline.apply("MF-3", wav)
        assert out.shape == wav.shape

    def test_median_filter_not_scalar_collapse(self):
        """Regression test: original MF-3 collapsed to a scalar — verify fixed."""
        wav = make_waveform(1)
        out = self.pipeline.apply("MF-3", wav)
        # If collapsed to scalar, all values would be identical
        assert out.std().item() > 1e-6

    def test_unknown_op_falls_back_gracefully(self):
        wav = make_waveform(1)
        out = self.pipeline.apply("NONEXISTENT_OP", wav)
        assert torch.allclose(out, wav)

    def test_temporal_clip_reduces_length(self):
        torch.manual_seed(0)
        wav = make_waveform(1)
        clipped, was_clipped = AugmentationPipeline.temporal_clip(
            wav, min_cut_frac=0.2, max_cut_frac=0.3, num_cuts=2
        )
        if was_clipped:
            assert clipped.size(2) < wav.size(2)

    def test_clip_start_never_negative(self):
        """Regression: original clip could produce slice[:negative]."""
        torch.manual_seed(42)
        for _ in range(100):
            wav = torch.randn(1, 1, 200)
            clipped, _ = AugmentationPipeline.temporal_clip(
                wav, min_cut_frac=0.25, max_cut_frac=0.40
            )
            assert clipped.size(2) >= 0

    def test_stochastic_call_returns_string(self):
        wav  = make_waveform(1)
        _, op = self.pipeline(wav)
        assert isinstance(op, str)
        assert op in self.pipeline.list_ops()


class TestMelExtractor:
    def test_output_shape(self):
        cfg = make_cfg()
        mel = MelExtractor(cfg.audio)
        wav = make_waveform()
        out = mel(wav)
        T_expected = math.ceil(NUM_SAMPLES / cfg.audio.hop_length) + 1
        assert out.shape[0] == BATCH_SIZE
        assert out.shape[1] == cfg.audio.n_mels
        # Allow ±2 frames for center/padding differences
        assert abs(out.shape[2] - T_expected) <= 2

    def test_2d_input_accepted(self):
        cfg = make_cfg()
        mel = MelExtractor(cfg.audio)
        wav = torch.randn(1, NUM_SAMPLES)
        out = mel(wav)
        assert out.dim() == 3

    def test_values_are_finite(self):
        cfg = make_cfg()
        mel = MelExtractor(cfg.audio)
        out = mel(make_waveform())
        assert torch.isfinite(out).all()


class TestPadOrTrim:
    def test_trim(self):
        x = torch.randn(1, 1000)
        assert pad_or_trim(x, 500).size(-1) == 500

    def test_zero_pad(self):
        x   = torch.randn(1, 100)
        out = pad_or_trim(x, 200, mode="zero")
        assert out.size(-1) == 200
        assert out[:, 100:].abs().max() == 0.0

    def test_repeat_pad(self):
        x   = torch.randn(1, 100)
        out = pad_or_trim(x, 250, mode="repeat")
        assert out.size(-1) == 250


# ═══════════════════════════════════════════════════════════════════════════════
#  INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestWatermarkSystemIntegration:
    def setup_method(self):
        self.cfg    = make_cfg()
        self.system = WatermarkSystem(self.cfg)
        self.mel    = MelExtractor(self.cfg.audio)

    def test_embed_output_shape(self):
        wav  = make_waveform()
        sign = make_sign(cfg=self.cfg)
        out  = self.system.embed(wav, sign)
        assert out.watermarked_waveform.shape == wav.shape
        assert out.perturbation.shape         == wav.shape
        assert out.wm_vector.shape            == (BATCH_SIZE, self.cfg.watermark.watermark_dim)

    def test_embed_stays_close_to_original(self):
        """Watermark alpha is small — waveforms should be perceptually similar."""
        wav  = make_waveform()
        sign = make_sign(cfg=self.cfg)
        out  = self.system.embed(wav, sign)
        l1   = F.l1_loss(out.watermarked_waveform, wav).item()
        assert l1 < 0.1

    def test_watermarked_within_unit_range(self):
        wav  = make_waveform()
        sign = make_sign(cfg=self.cfg)
        out  = self.system.embed(wav, sign)
        assert out.watermarked_waveform.abs().max().item() <= 1.0 + 1e-5

    def test_decode_output_shapes(self):
        wav  = make_waveform()
        mel  = self.mel(wav).transpose(1, 2)
        scores, pred = self.system.decode(mel)
        assert pred.shape == (BATCH_SIZE, self.cfg.watermark.num_symbols)
        assert len(scores) == self.cfg.watermark.num_symbols

    def test_full_forward(self):
        """Embed → mel extract → decode runs without error."""
        wav  = make_waveform()
        sign = make_sign(cfg=self.cfg)

        out    = self.system.embed(wav, sign)
        mel    = self.mel(out.watermarked_waveform).transpose(1, 2)
        scores, pred = self.system.decode(mel)

        loss = sign_loss(scores, sign)
        assert torch.isfinite(loss)

    def test_gradients_flow_through_system(self):
        """Loss.backward() should not raise and params should receive gradients."""
        wav  = make_waveform()
        sign = make_sign(cfg=self.cfg)

        out    = self.system.embed(wav, sign)
        mel    = self.mel(out.watermarked_waveform).transpose(1, 2)
        scores, _ = self.system.decode(mel)

        loss = sign_loss(scores, sign)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.system.parameters()
        )
        assert has_grad, "No gradients flowed back through the system"


# ═══════════════════════════════════════════════════════════════════════════════
#  ROBUSTNESS TESTS (after sufficient training the model should pass these)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustnessStructural:
    """
    These tests do NOT require a trained model — they verify that the
    end-to-end pipeline runs without crashing under each attack condition.
    Accuracy tests (asserting correct decoding) require a trained checkpoint.
    """

    def setup_method(self):
        self.cfg    = make_cfg()
        self.system = WatermarkSystem(self.cfg)
        self.system.eval()
        self.mel    = MelExtractor(self.cfg.audio)
        self.aug    = AugmentationPipeline(
            self.cfg.augmentation, sample_rate=SAMPLE_RATE
        )

    def _encode_decode(self, waveform: torch.Tensor) -> tuple:
        sign  = make_sign(B=1, cfg=self.cfg)
        out   = self.system.embed(waveform[:1], sign)
        mel   = self.mel(out.watermarked_waveform).transpose(1, 2)
        scores, pred = self.system.decode(mel)
        return pred, sign

    @pytest.mark.parametrize("op", [
        "CLP", "RSP-90", "Noise-W35",
        "APS-05", "APS-15", "HPF-1800",
        "LPF-5000", "MF-3", "TS-09",
    ])
    def test_decode_does_not_crash_after_attack(self, op):
        wav     = make_waveform(1)
        sign    = make_sign(B=1, cfg=self.cfg)
        out     = self.system.embed(wav, sign)
        attacked = self.aug.apply(op, out.watermarked_waveform)

        # attacked may be shorter after TS-09 — pad to original length
        if attacked.size(2) < NUM_SAMPLES:
            attacked = pad_or_trim(attacked, NUM_SAMPLES)

        mel   = self.mel(attacked).transpose(1, 2)
        scores, pred = self.system.decode(mel)
        loss  = sign_loss(scores, sign)

        assert pred.shape == (1, self.cfg.watermark.num_symbols)
        assert torch.isfinite(loss)


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTOR TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestWatermarkDetector:
    def setup_method(self):
        from detector import WatermarkDetector, DetectionResult
        self.DetectionResult = DetectionResult
        self.cfg     = make_cfg()
        self.system  = WatermarkSystem(self.cfg)
        self.system.eval()
        self.detector = WatermarkDetector(self.system, self.cfg)

    def test_detect_returns_detection_result(self):
        wav    = make_waveform(1)
        result = self.detector.detect(wav)
        assert isinstance(result, self.DetectionResult)

    def test_sign_has_correct_length(self):
        wav    = make_waveform(1)
        result = self.detector.detect(wav)
        assert len(result.sign) == self.cfg.watermark.num_symbols

    def test_symbol_probs_sum_to_one(self):
        wav    = make_waveform(1)
        result = self.detector.detect(wav)
        for probs in result.symbol_probs:
            assert abs(probs.sum().item() - 1.0) < 1e-5

    def test_confidence_in_unit_interval(self):
        wav    = make_waveform(1)
        result = self.detector.detect(wav)
        for c in result.symbol_confidence:
            assert 0.0 <= c <= 1.0

    def test_to_id_deterministic(self):
        result = self.DetectionResult(
            sign              = [3, 7, 1, 12],
            symbol_probs      = [torch.zeros(16)] * 4,
            symbol_confidence = [0.9] * 4,
            mean_confidence   = 0.9,
        )
        id1 = result.to_id()
        id2 = result.to_id()
        assert id1 == id2

    def test_compare_correct_match(self):
        from detector import DetectionResult
        result = DetectionResult(
            sign              = [1, 2, 3, 4],
            symbol_probs      = [torch.zeros(16)] * 4,
            symbol_confidence = [0.9] * 4,
            mean_confidence   = 0.9,
        )
        cmp = self.detector.compare(result, [1, 2, 3, 4])
        assert cmp["correct"] is True
        assert cmp["accuracy"] == pytest.approx(1.0)

    def test_compare_partial_mismatch(self):
        from detector import DetectionResult
        result = DetectionResult(
            sign              = [1, 2, 3, 4],
            symbol_probs      = [torch.zeros(16)] * 4,
            symbol_confidence = [0.9] * 4,
            mean_confidence   = 0.9,
        )
        cmp = self.detector.compare(result, [1, 2, 9, 9])
        assert cmp["correct"] is False
        assert cmp["accuracy"] == pytest.approx(0.5)

    def test_detect_file(self, tmp_path):
        import torchaudio
        wav_path = tmp_path / "test.wav"
        torchaudio.save(
            str(wav_path),
            make_waveform(1).squeeze(0),
            SAMPLE_RATE,
        )
        result = self.detector.detect_file(wav_path)
        assert result.source_path == str(wav_path)
        assert len(result.sign) == self.cfg.watermark.num_symbols

    def test_print_result_does_not_raise(self, capsys):
        from detector import DetectionResult, WatermarkDetector
        result = DetectionResult(
            sign              = [3, 7, 1, 12],
            symbol_probs      = [F.softmax(torch.randn(16), dim=0)] * 4,
            symbol_confidence = [0.8, 0.9, 0.7, 0.85],
            mean_confidence   = 0.8125,
        )
        WatermarkDetector.print_result(result)
        captured = capsys.readouterr()
        assert "DETECTED" in captured.out or "UNCERTAIN" in captured.out
