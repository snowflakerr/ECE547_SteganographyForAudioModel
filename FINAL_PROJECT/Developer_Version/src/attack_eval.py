"""
attack_eval.py
==============
Run a battery of distortion attacks against a watermarked audio file,
extract the payload after each attack, and report robustness.

For each attack we save:
  - <attack>.wav             — the attacked audio
  - <attack>_spectrogram.png — log-magnitude spectrogram
  - results.csv              — payload, BER, accuracy, SNR per attack
  - summary.png              — bar chart of bit accuracy across attacks

The watermark scheme matched here is the spread-spectrum FramewiseWatermarker
from generate_demo.py: BPSK symbols ride a PN-sequence carrier in the STFT
magnitude. Extraction correlates the (attacked − reference) residual with the
same PN matrix; if you don't have the clean reference, we fall back to a
zero-magnitude reference, which works less well but still recovers most bits
because the PN signal is uncorrelated with natural speech magnitude.

Usage
-----
    python attack_eval.py \
        --watermarked path/to/watermarked.wav \
        --clean       path/to/clean.wav \
        --payload     42 \
        --key         traceable_speech_demo \
        --alpha       0.01 \
        --out_dir     attack_results

Short-clip demo (≈0.3 s):
    python attack_eval.py --clip_seconds 0.3 ...
"""

from __future__ import annotations

import argparse
import csv
import io
import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy.signal import (
    butter,
    fftconvolve,
    lfilter,
    resample_poly,
    stft,
)

# Reuse the embedder so the PN matrix stays consistent.
from generate_demo import FramewiseWatermarker


# ──────────────────────────────────────────────────────────────────────────────
# Extraction (companion to FramewiseWatermarker.embed)
# ──────────────────────────────────────────────────────────────────────────────


def _stft_mag(audio: np.ndarray, sr: int, nperseg: int, noverlap: int) -> np.ndarray:
    _, _, Z = stft(audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
                   boundary="zeros", padded=True)
    return np.abs(Z).astype(np.float32)  # [n_bins, n_frames]


def extract_payload(
    attacked_audio: np.ndarray,
    sample_rate: int,
    watermarker: FramewiseWatermarker,
    reference_audio: Optional[np.ndarray] = None,
    nperseg: int = 1024,
    noverlap: int = 768,
    embed_frame_count: Optional[int] = None,
) -> tuple[int, np.ndarray]:
    """Recover the BPSK payload from a (possibly attacked) waveform.

    Critical detail: the PN matrix used at embed time was sized to the
    *embed-time* frame count.  Because numpy's RNG fills row by row,
    PN[k] when sampled with n_frames=N differs from PN[k] when sampled
    with n_frames=N±1.  Resample/echo/reverb attacks *do* change the
    audio length and therefore the frame count, so we MUST size the
    PN matrix to whatever the embedder used, not to whatever the
    attacked audio gives us, then align the attacked frames to that.

    Strategy
    --------
    1. Use ``embed_frame_count`` (or fall back to the watermarked
       reference's frame count, or the attacked count as last resort)
       to regenerate the same PN matrix the embedder used.
    2. Truncate / zero-pad the attacked magnitude (and reference, if
       provided) to that frame count.
    3. residual = attacked_mag − reference_mag (else attacked_mag).
       The latter still works because PN is uncorrelated with speech.
    4. Energy-weight per frame so quiet frames (where the magnitude
       was clipped at zero in embed) don't dominate.
    5. Per-frame correlation summed across freq bins, then averaged
       over the frames assigned to each bit (round-robin) → sign → bit.
    """
    mag_att = _stft_mag(attacked_audio, sample_rate, nperseg, noverlap)

    # Decide frame count for the PN matrix — must match the embedder.
    if embed_frame_count is not None:
        n_frames_embed = int(embed_frame_count)
    elif reference_audio is not None:
        n_frames_embed = _stft_mag(reference_audio, sample_rate, nperseg, noverlap).shape[1]
    else:
        n_frames_embed = mag_att.shape[1]

    n_bins = mag_att.shape[0]

    def _fit(mag: np.ndarray, target_frames: int) -> np.ndarray:
        cur = mag.shape[1]
        if cur == target_frames:
            return mag
        if cur > target_frames:
            return mag[:, :target_frames]
        pad = np.zeros((mag.shape[0], target_frames - cur), dtype=mag.dtype)
        return np.concatenate([mag, pad], axis=1)

    mag_att = _fit(mag_att, n_frames_embed)

    if reference_audio is not None:
        mag_ref = _stft_mag(reference_audio, sample_rate, nperseg, noverlap)
        mag_ref = _fit(mag_ref, n_frames_embed)
        residual = mag_att - mag_ref
    else:
        residual = mag_att

    # Same PN matrix the embedder produced.
    pn = watermarker._pn_matrix(n_frames_embed, n_bins)  # [n_frames, n_bins]

    # Matched-filter weighting.
    #
    # The embedder added:  alpha * frame_rms[t] * pn[t, b]   to magnitude[b, t]
    # (perceptual weighting).  So the *expected* watermark amplitude in each
    # frame is proportional to that frame's RMS magnitude.  The optimal linear
    # decoder weights each frame by its expected signal strength — silent
    # frames carry no watermark and only contribute noise, so they get ~0
    # weight; loud frames carry strong watermark and get full weight.
    #
    # When we have the clean reference, frame_rms_clean is the right weight.
    # Without it, fall back to attacked-magnitude RMS (still better than nothing
    # — speech amplitude structure dominates the watermark amplitude).
    if reference_audio is not None:
        weight_source = mag_ref
    else:
        weight_source = mag_att
    frame_rms = np.sqrt((weight_source ** 2).mean(axis=0) + 1e-12)  # [n_frames]

    # Per-frame correlation across freq bins.
    frame_corr = np.einsum("fb,bf->f", pn, residual)  # [n_frames]

    # Weighted decode: bit[k] = sign(sum_t (w_t * corr_t) for t in bit-k frames)
    bits = np.zeros(watermarker.payload_bits, dtype=np.int32)
    for bit_idx in range(watermarker.payload_bits):
        mask = (np.arange(n_frames_embed) % watermarker.payload_bits) == bit_idx
        if not mask.any():
            bits[bit_idx] = 0
            continue
        w = frame_rms[mask]
        c = frame_corr[mask]
        score = float((w * c).sum())
        bits[bit_idx] = 1 if score > 0 else 0

    payload_int = int(sum(int(b) << i for i, b in enumerate(bits)))
    return payload_int, bits


def payload_to_bits(payload: int, n_bits: int) -> np.ndarray:
    return np.array([(payload >> i) & 1 for i in range(n_bits)], dtype=np.int32)


def bit_error_rate(true_bits: np.ndarray, recovered_bits: np.ndarray) -> float:
    return float(np.mean(true_bits != recovered_bits))


# ──────────────────────────────────────────────────────────────────────────────
# Attacks
# ──────────────────────────────────────────────────────────────────────────────


def _normalize_peak(x: np.ndarray, target: float = 0.95) -> np.ndarray:
    peak = float(np.abs(x).max()) if x.size else 0.0
    if peak > 0:
        x = x / peak * target
    return x.astype(np.float32)


def attack_additive_noise(audio: np.ndarray, sr: int, snr_db: float = 20.0) -> np.ndarray:
    """White Gaussian noise at a given SNR."""
    sig_power = float(np.mean(audio ** 2)) + 1e-12
    snr_lin = 10 ** (snr_db / 10.0)
    noise_std = math.sqrt(sig_power / snr_lin)
    noise = np.random.randn(len(audio)).astype(np.float32) * noise_std
    return (audio + noise).astype(np.float32)


def attack_mp3(audio: np.ndarray, sr: int, bitrate: str = "64k") -> np.ndarray:
    """MP3 codec round-trip via ffmpeg. Falls back to AAC if MP3 missing."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        in_wav = td_path / "in.wav"
        comp = td_path / "out.mp3"
        out_wav = td_path / "out.wav"

        sf.write(in_wav, audio, sr, subtype="PCM_16")
        # Encode
        enc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(in_wav), "-b:a", bitrate, str(comp)],
            capture_output=True,
        )
        if enc.returncode != 0:
            raise RuntimeError(f"ffmpeg encode failed: {enc.stderr.decode()}")
        # Decode back to wav at original sr
        dec = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-i", str(comp), "-ar", str(sr), str(out_wav)],
            capture_output=True,
        )
        if dec.returncode != 0:
            raise RuntimeError(f"ffmpeg decode failed: {dec.stderr.decode()}")

        decoded, decoded_sr = sf.read(out_wav)
    if decoded.ndim > 1:
        decoded = decoded.mean(axis=1)
    if decoded_sr != sr:
        from math import gcd
        g = gcd(sr, decoded_sr)
        decoded = resample_poly(decoded, sr // g, decoded_sr // g)
    # Pad/trim to original length
    if len(decoded) < len(audio):
        decoded = np.pad(decoded, (0, len(audio) - len(decoded)))
    else:
        decoded = decoded[: len(audio)]
    return decoded.astype(np.float32)


def attack_resample(audio: np.ndarray, sr: int, intermediate_sr: int = 8000) -> np.ndarray:
    """Down → up resample, keeping output length aligned."""
    from math import gcd
    g1 = gcd(sr, intermediate_sr)
    down = resample_poly(audio, intermediate_sr // g1, sr // g1)
    g2 = gcd(intermediate_sr, sr)
    up = resample_poly(down, sr // g2, intermediate_sr // g2)
    if len(up) < len(audio):
        up = np.pad(up, (0, len(audio) - len(up)))
    else:
        up = up[: len(audio)]
    return up.astype(np.float32)


def attack_lowpass(audio: np.ndarray, sr: int, cutoff_hz: float = 4000.0) -> np.ndarray:
    nyq = sr / 2
    cutoff = min(cutoff_hz, nyq * 0.99)
    b, a = butter(N=6, Wn=cutoff / nyq, btype="low")
    return lfilter(b, a, audio).astype(np.float32)


def attack_highpass(audio: np.ndarray, sr: int, cutoff_hz: float = 200.0) -> np.ndarray:
    nyq = sr / 2
    b, a = butter(N=6, Wn=cutoff_hz / nyq, btype="high")
    return lfilter(b, a, audio).astype(np.float32)


def attack_bandpass(audio: np.ndarray, sr: int,
                    low_hz: float = 300.0, high_hz: float = 3400.0) -> np.ndarray:
    """Telephone-band bandpass."""
    nyq = sr / 2
    high_hz = min(high_hz, nyq * 0.99)
    b, a = butter(N=6, Wn=[low_hz / nyq, high_hz / nyq], btype="band")
    return lfilter(b, a, audio).astype(np.float32)


def attack_echo(audio: np.ndarray, sr: int,
                delay_ms: float = 120.0, decay: float = 0.5) -> np.ndarray:
    delay_samples = int(sr * delay_ms / 1000.0)
    out = np.copy(audio).astype(np.float32)
    if delay_samples < len(out):
        out[delay_samples:] += decay * audio[:-delay_samples]
    return _normalize_peak(out, target=0.95)


def attack_reverb(audio: np.ndarray, sr: int,
                  rt60_seconds: float = 0.4) -> np.ndarray:
    """Synthetic reverb: convolve with an exponentially decaying noise IR."""
    ir_len = int(rt60_seconds * sr)
    if ir_len < 8:
        return audio.astype(np.float32)
    t = np.arange(ir_len) / sr
    # Decay envelope so amplitude reaches 1e-3 of initial at rt60
    envelope = np.exp(-6.9 * t / rt60_seconds)
    ir = np.random.randn(ir_len).astype(np.float32) * envelope
    ir[0] = 1.0  # direct path
    wet = fftconvolve(audio, ir, mode="full")[: len(audio)]
    # Mix dry + wet 50/50
    out = 0.5 * audio + 0.5 * wet
    return _normalize_peak(out, target=0.95)


def attack_dropout(audio: np.ndarray, sr: int,
                   drop_prob: float = 0.05,
                   chunk_ms: float = 20.0,
                   seed: int = 0) -> np.ndarray:
    """Packet-loss style: zero out random fixed-length chunks."""
    rng = np.random.default_rng(seed)
    chunk = max(1, int(sr * chunk_ms / 1000.0))
    out = audio.copy().astype(np.float32)
    n_chunks = len(out) // chunk
    drop_mask = rng.random(n_chunks) < drop_prob
    for i, drop in enumerate(drop_mask):
        if drop:
            out[i * chunk : (i + 1) * chunk] = 0.0
    return out


# Registry: name → (callable, label-for-plot)
ATTACK_REGISTRY: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "clean":            lambda a, sr: a.astype(np.float32),
    "noise_30dB":       lambda a, sr: attack_additive_noise(a, sr, snr_db=30),
    "noise_20dB":       lambda a, sr: attack_additive_noise(a, sr, snr_db=20),
    "noise_10dB":       lambda a, sr: attack_additive_noise(a, sr, snr_db=10),
    "mp3_64k":          lambda a, sr: attack_mp3(a, sr, bitrate="64k"),
    "mp3_32k":          lambda a, sr: attack_mp3(a, sr, bitrate="32k"),
    "resample_8k":      lambda a, sr: attack_resample(a, sr, intermediate_sr=8000),
    "resample_16k":     lambda a, sr: attack_resample(a, sr, intermediate_sr=16000),
    "lowpass_4k":       lambda a, sr: attack_lowpass(a, sr, cutoff_hz=4000),
    "highpass_200":     lambda a, sr: attack_highpass(a, sr, cutoff_hz=200),
    "bandpass_phone":   lambda a, sr: attack_bandpass(a, sr, 300, 3400),
    "echo_120ms":       lambda a, sr: attack_echo(a, sr, delay_ms=120, decay=0.5),
    "reverb_400ms":     lambda a, sr: attack_reverb(a, sr, rt60_seconds=0.4),
    "dropout_5pct":     lambda a, sr: attack_dropout(a, sr, drop_prob=0.05),
    "dropout_15pct":    lambda a, sr: attack_dropout(a, sr, drop_prob=0.15),
}


# ──────────────────────────────────────────────────────────────────────────────
# Metrics & I/O
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class AttackResult:
    name: str
    payload_recovered: int
    bit_accuracy: float
    bit_error_rate: float
    snr_vs_watermarked_db: float
    success: bool
    audio_path: str
    spectrogram_path: str


def snr_db(reference: np.ndarray, distorted: np.ndarray) -> float:
    L = min(len(reference), len(distorted))
    ref = reference[:L]
    dist = distorted[:L]
    noise = dist - ref
    sig_p = float(np.mean(ref ** 2)) + 1e-12
    n_p = float(np.mean(noise ** 2)) + 1e-12
    return 10.0 * math.log10(sig_p / n_p)


def save_spectrogram_png(audio: np.ndarray, sr: int, path: Path, title: str) -> None:
    f, t, Z = stft(audio, fs=sr, nperseg=1024, noverlap=768)
    mag_db = 20 * np.log10(np.abs(Z) + 1e-8)

    fig, ax = plt.subplots(figsize=(7, 3.2), dpi=120)
    im = ax.pcolormesh(t, f, mag_db, shading="gouraud", cmap="magma",
                       vmin=mag_db.max() - 80, vmax=mag_db.max())
    ax.set_ylabel("Frequency (Hz)")
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_summary_chart(results: list[AttackResult], path: Path) -> None:
    names = [r.name for r in results]
    accs = [r.bit_accuracy * 100 for r in results]
    colors = ["#22c55e" if r.success else "#ef4444" for r in results]

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
    bars = ax.bar(names, accs, color=colors, edgecolor="#111")
    ax.axhline(50, color="#888", linestyle="--", linewidth=1, label="Chance (50%)")
    ax.axhline(90, color="#22c55e", linestyle=":", linewidth=1, label="Success threshold (90%)")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Bit accuracy (%)")
    ax.set_title("Watermark robustness under distortion attacks")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.legend(loc="lower left")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.0f}%", ha="center", fontsize=8)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_waveform_compare(clean_wm: np.ndarray, attacked: np.ndarray,
                          sr: int, path: Path, attack_name: str) -> None:
    L = min(len(clean_wm), len(attacked))
    t = np.arange(L) / sr
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.6), dpi=120, sharex=True)
    axes[0].plot(t, clean_wm[:L], color="#3b82f6", linewidth=0.7)
    axes[0].set_title("Watermarked (before attack)")
    axes[0].set_ylabel("Amplitude")
    axes[1].plot(t, attacked[:L], color="#ef4444", linewidth=0.7)
    axes[1].set_title(f"After attack: {attack_name}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────


def run_evaluation(
    watermarked_path: Path,
    clean_path: Optional[Path],
    payload: int,
    key: str,
    alpha: float,
    out_dir: Path,
    clip_seconds: Optional[float] = None,
    attacks: Optional[list[str]] = None,
    success_threshold: float = 0.9,
) -> list[AttackResult]:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(exist_ok=True)
    (out_dir / "spectrograms").mkdir(exist_ok=True)
    (out_dir / "waveforms").mkdir(exist_ok=True)

    # Load watermarked audio
    wm_audio, sr = sf.read(watermarked_path)
    if wm_audio.ndim > 1:
        wm_audio = wm_audio.mean(axis=1)
    wm_audio = wm_audio.astype(np.float32)

    clean_audio = None
    if clean_path is not None and clean_path.exists():
        clean_audio, clean_sr = sf.read(clean_path)
        if clean_audio.ndim > 1:
            clean_audio = clean_audio.mean(axis=1)
        if clean_sr != sr:
            from math import gcd
            g = gcd(sr, clean_sr)
            clean_audio = resample_poly(clean_audio, sr // g, clean_sr // g)
        clean_audio = clean_audio.astype(np.float32)

    # Optional clipping for "short clip" demo
    if clip_seconds is not None:
        n = int(clip_seconds * sr)
        wm_audio = wm_audio[:n]
        if clean_audio is not None:
            clean_audio = clean_audio[: len(wm_audio)]
        print(f"  Clipped to {clip_seconds}s ({len(wm_audio)} samples)")

    # Reusable watermarker (only needs the secret key + bits to rebuild PN matrix)
    wm = FramewiseWatermarker(secret_key=key, alpha=alpha, payload_bits=32, perceptual=True)
    true_bits = payload_to_bits(payload, wm.payload_bits)

    # Capture the embed-time STFT frame count from the watermarked file.
    # Every extraction must use a PN matrix sized to this so it lines up
    # with what was embedded — even after attacks that change audio length.
    embed_frame_count = _stft_mag(wm_audio, sr, nperseg=1024, noverlap=768).shape[1]

    # Save the "before" reference (watermarked, no attack applied yet)
    sf.write(out_dir / "audio" / "00_watermarked_before.wav", wm_audio, sr, subtype="PCM_16")
    save_spectrogram_png(wm_audio, sr, out_dir / "spectrograms" / "00_watermarked_before.png",
                         "Watermarked — before any attack")

    # Pick attacks
    chosen = attacks or list(ATTACK_REGISTRY.keys())

    results: list[AttackResult] = []
    print(f"\nRunning {len(chosen)} attacks on {len(wm_audio)/sr:.2f}s clip...\n")
    print(f"{'Attack':<18} {'BER':>7} {'Acc':>7} {'SNR (dB)':>10} {'Result':>10}")
    print("-" * 60)

    for name in chosen:
        if name not in ATTACK_REGISTRY:
            print(f"  Skipping unknown attack: {name}")
            continue
        try:
            attacked = ATTACK_REGISTRY[name](wm_audio, sr)
        except Exception as exc:
            print(f"  {name:<18} FAILED: {exc}")
            continue

        # Recover payload — pin PN matrix to embed-time frame count.
        recovered_int, recovered_bits = extract_payload(
            attacked, sr, wm,
            reference_audio=clean_audio,
            embed_frame_count=embed_frame_count,
        )
        ber = bit_error_rate(true_bits, recovered_bits)
        acc = 1.0 - ber
        snr = snr_db(wm_audio, attacked)
        ok = acc >= success_threshold

        # Persist outputs
        audio_path = out_dir / "audio" / f"{name}.wav"
        spec_path = out_dir / "spectrograms" / f"{name}.png"
        wf_path = out_dir / "waveforms" / f"{name}.png"
        sf.write(audio_path, attacked, sr, subtype="PCM_16")
        save_spectrogram_png(attacked, sr, spec_path,
                             f"After: {name}  (acc={acc*100:.0f}%, SNR={snr:.1f} dB)")
        save_waveform_compare(wm_audio, attacked, sr, wf_path, name)

        results.append(AttackResult(
            name=name,
            payload_recovered=recovered_int,
            bit_accuracy=acc,
            bit_error_rate=ber,
            snr_vs_watermarked_db=snr,
            success=ok,
            audio_path=str(audio_path),
            spectrogram_path=str(spec_path),
        ))

        flag = "✓ pass" if ok else "✗ fail"
        print(f"{name:<18} {ber:>7.3f} {acc*100:>6.1f}% {snr:>10.2f} {flag:>10}")

    # CSV
    csv_path = out_dir / "results.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["attack", "bit_accuracy", "bit_error_rate", "snr_db",
                         "payload_true", "payload_recovered", "success"])
        for r in results:
            writer.writerow([r.name, f"{r.bit_accuracy:.4f}",
                             f"{r.bit_error_rate:.4f}",
                             f"{r.snr_vs_watermarked_db:.2f}",
                             payload, r.payload_recovered, int(r.success)])

    save_summary_chart(results, out_dir / "summary.png")

    # Console summary
    n_pass = sum(r.success for r in results)
    mean_acc = float(np.mean([r.bit_accuracy for r in results])) if results else 0.0
    print("-" * 60)
    print(f"Summary: {n_pass}/{len(results)} attacks survived (≥{success_threshold*100:.0f}% accuracy)")
    print(f"Mean bit accuracy across all attacks: {mean_acc*100:.1f}%")
    print(f"\nOutputs in: {out_dir.resolve()}")
    print(f"  audio/         attacked WAVs (one per attack)")
    print(f"  spectrograms/  log-magnitude spectrograms")
    print(f"  waveforms/     before/after waveform comparisons")
    print(f"  results.csv    machine-readable per-attack metrics")
    print(f"  summary.png    bar chart of bit accuracy")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Alpha sweep — embed at multiple strengths, attack each, chart trade-offs
# ──────────────────────────────────────────────────────────────────────────────


def _embed_at_alpha(
    clean_audio: np.ndarray,
    sr: int,
    payload: int,
    key: str,
    alpha: float,
    nperseg: int = 1024,
    noverlap: int = 768,
) -> np.ndarray:
    """Embed the watermark at a given alpha, mirroring generate_demo.apply_watermark.

    Kept inline so the sweep doesn't import the espeak-dependent module path
    and so we know exactly what's happening at each step.
    """
    wm = FramewiseWatermarker(secret_key=key, alpha=alpha,
                              payload_bits=32, perceptual=True)
    _, _, Z = stft(clean_audio, fs=sr, nperseg=nperseg, noverlap=noverlap,
                   boundary="zeros", padded=True)
    mag = np.abs(Z)
    phase = np.angle(Z)

    wm_mag = wm.embed(mag, payload)
    Z_wm = wm_mag * np.exp(1j * phase)

    from scipy.signal import istft
    _, audio_wm = istft(Z_wm, fs=sr, nperseg=nperseg, noverlap=noverlap, boundary=True)
    audio_wm = audio_wm[: len(clean_audio)]
    if len(audio_wm) < len(clean_audio):
        audio_wm = np.pad(audio_wm, (0, len(clean_audio) - len(audio_wm)))

    # Match peak so loudness is identical (same step generate_demo does)
    orig_peak = float(np.abs(clean_audio).max())
    wm_peak = float(np.abs(audio_wm).max())
    if wm_peak > 0:
        audio_wm = audio_wm * (orig_peak / wm_peak)
    return audio_wm.astype(np.float32)


def run_alpha_sweep(
    clean_path: Path,
    payload: int,
    key: str,
    alphas: list[float],
    out_dir: Path,
    clip_seconds: Optional[float] = None,
    attacks: Optional[list[str]] = None,
    success_threshold: float = 0.9,
) -> dict:
    """Sweep embedding strength: at each alpha, embed → attack → extract → save.

    Outputs in ``out_dir``:
      perceptibility/alpha_{a}.wav   — watermarked clip at each alpha (for listening)
      perceptibility/clean.wav       — the clean reference (for A/B)
      alpha_{a}/                     — full attack-eval folder for that alpha
      sweep_heatmap.png              — alpha × attack accuracy grid
      sweep_tradeoff.png             — mean attack accuracy vs perceptual SNR
      sweep_results.csv              — flat table for all (alpha, attack) cells
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    listen_dir = out_dir / "perceptibility"
    listen_dir.mkdir(exist_ok=True)

    clean_audio, sr = sf.read(clean_path)
    if clean_audio.ndim > 1:
        clean_audio = clean_audio.mean(axis=1)
    clean_audio = clean_audio.astype(np.float32)
    if clip_seconds is not None:
        clean_audio = clean_audio[: int(clip_seconds * sr)]

    sf.write(listen_dir / "clean.wav", clean_audio, sr, subtype="PCM_16")

    chosen = attacks or list(ATTACK_REGISTRY.keys())

    # Per-alpha results: {alpha: {attack_name: AttackResult}}
    sweep: dict[float, dict[str, AttackResult]] = {}
    perceptual_snrs: dict[float, float] = {}

    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"  Alpha = {alpha:.4f}")
        print(f"{'='*60}")

        # 1. Embed at this strength
        wm_audio = _embed_at_alpha(clean_audio, sr, payload, key, alpha)

        # 2. Save listening clip & clean reference for this alpha
        wm_listen_path = listen_dir / f"alpha_{alpha:.3f}.wav"
        sf.write(wm_listen_path, wm_audio, sr, subtype="PCM_16")

        # Perceptual SNR: clean vs watermarked (higher = less audible)
        psnr = snr_db(clean_audio, wm_audio)
        perceptual_snrs[alpha] = psnr
        print(f"  Watermarked clip saved → {wm_listen_path}")
        print(f"  Perceptual SNR (clean vs wm): {psnr:.2f} dB  "
              f"(higher = less audible; ~40 dB is typically inaudible)")

        # 3. Persist a temp watermarked WAV and run the full eval
        alpha_dir = out_dir / f"alpha_{alpha:.3f}"
        alpha_dir.mkdir(exist_ok=True)
        tmp_wm = alpha_dir / "_watermarked.wav"
        tmp_clean = alpha_dir / "_clean.wav"
        sf.write(tmp_wm, wm_audio, sr, subtype="PCM_16")
        sf.write(tmp_clean, clean_audio, sr, subtype="PCM_16")

        results = run_evaluation(
            watermarked_path=tmp_wm,
            clean_path=tmp_clean,
            payload=payload,
            key=key,
            alpha=alpha,
            out_dir=alpha_dir,
            clip_seconds=None,    # already clipped
            attacks=chosen,
            success_threshold=success_threshold,
        )
        sweep[alpha] = {r.name: r for r in results}

    # ── Build aggregate outputs ──────────────────────────────────────────────

    # Heatmap: rows = attacks (in order), cols = alphas
    attack_names = chosen
    acc_grid = np.zeros((len(attack_names), len(alphas)), dtype=np.float32)
    for j, a in enumerate(alphas):
        for i, name in enumerate(attack_names):
            r = sweep[a].get(name)
            acc_grid[i, j] = r.bit_accuracy * 100 if r else 0.0

    fig, ax = plt.subplots(figsize=(1.2 + 1.0 * len(alphas), 0.35 * len(attack_names) + 1.5),
                           dpi=120)
    im = ax.imshow(acc_grid, aspect="auto", cmap="RdYlGn", vmin=50, vmax=100)
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.3f}" for a in alphas])
    ax.set_yticks(range(len(attack_names)))
    ax.set_yticklabels(attack_names)
    ax.set_xlabel("Embedding strength (alpha)")
    ax.set_title("Bit accuracy (%) — alpha sweep")
    for i in range(len(attack_names)):
        for j in range(len(alphas)):
            ax.text(j, i, f"{acc_grid[i, j]:.0f}", ha="center", va="center",
                    fontsize=8, color="black")
    fig.colorbar(im, ax=ax, label="Bit accuracy (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_heatmap.png")
    plt.close(fig)

    # Trade-off chart: mean attack accuracy vs perceptual SNR per alpha
    mean_accs = [
        float(np.mean([r.bit_accuracy for r in sweep[a].values()])) * 100 for a in alphas
    ]
    psnr_values = [perceptual_snrs[a] for a in alphas]

    fig, ax1 = plt.subplots(figsize=(8, 4.2), dpi=120)
    ax1.plot(alphas, mean_accs, "o-", color="#22c55e", linewidth=2,
             markersize=8, label="Mean attack accuracy")
    ax1.set_xlabel("Embedding strength (alpha)")
    ax1.set_ylabel("Mean bit accuracy across attacks (%)", color="#22c55e")
    ax1.set_ylim(40, 105)
    ax1.tick_params(axis="y", labelcolor="#22c55e")
    ax1.axhline(90, color="#22c55e", linestyle=":", linewidth=1, alpha=0.6)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(alphas, psnr_values, "s--", color="#3b82f6", linewidth=2,
             markersize=7, label="Perceptual SNR")
    ax2.set_ylabel("Perceptual SNR (dB) — higher = less audible", color="#3b82f6")
    ax2.tick_params(axis="y", labelcolor="#3b82f6")
    ax2.axhline(40, color="#3b82f6", linestyle=":", linewidth=1, alpha=0.6)
    # 40 dB rule-of-thumb annotation
    ax2.text(alphas[0], 40.5, "  ~40 dB ≈ inaudibility threshold",
             color="#3b82f6", fontsize=8, va="bottom")

    plt.title("Robustness vs perceptibility trade-off")
    fig.tight_layout()
    fig.savefig(out_dir / "sweep_tradeoff.png")
    plt.close(fig)

    # Flat CSV
    with (out_dir / "sweep_results.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["alpha", "attack", "bit_accuracy", "ber", "snr_db",
                    "perceptual_snr_db", "success"])
        for a in alphas:
            for name in attack_names:
                r = sweep[a].get(name)
                if r is None:
                    continue
                w.writerow([f"{a:.4f}", name,
                            f"{r.bit_accuracy:.4f}",
                            f"{r.bit_error_rate:.4f}",
                            f"{r.snr_vs_watermarked_db:.2f}",
                            f"{perceptual_snrs[a]:.2f}",
                            int(r.success)])

    # Console summary
    print(f"\n{'='*60}")
    print("  Alpha sweep summary")
    print(f"{'='*60}")
    print(f"  {'alpha':<8} {'mean_acc':>10} {'#pass':>8} {'perc_SNR':>12}")
    for a, mean_acc in zip(alphas, mean_accs):
        n_pass = sum(r.success for r in sweep[a].values())
        print(f"  {a:<8.4f} {mean_acc:>9.1f}% {n_pass:>3}/{len(attack_names):<3} "
              f"{perceptual_snrs[a]:>10.2f} dB")
    print(f"\n  Listen to perceptibility samples in: {(out_dir / 'perceptibility').resolve()}")
    print(f"  Heatmap:   {out_dir / 'sweep_heatmap.png'}")
    print(f"  Trade-off: {out_dir / 'sweep_tradeoff.png'}")

    return {
        "alphas": alphas,
        "sweep": sweep,
        "perceptual_snrs": perceptual_snrs,
        "mean_accs": mean_accs,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--watermarked", type=Path, default=None,
                    help="Path to the watermarked WAV to attack. "
                         "Required unless --alpha_sweep is set.")
    ap.add_argument("--clean", type=Path, default=None,
                    help="Optional path to the clean (un-watermarked) reference WAV. "
                         "Strongly recommended — gives much better extraction accuracy.")
    ap.add_argument("--payload", type=int, default=42,
                    help="The integer payload that was embedded (default: 42).")
    ap.add_argument("--key", type=str, default="traceable_speech_demo",
                    help="Secret key used during embedding (default: traceable_speech_demo).")
    ap.add_argument("--alpha", type=float, default=0.01,
                    help="Embedding strength used at embed time (default: 0.01).")
    ap.add_argument("--out_dir", type=Path, default=Path("attack_results"),
                    help="Directory for all outputs (default: attack_results).")
    ap.add_argument("--clip_seconds", type=float, default=None,
                    help="If set, truncate to this many seconds first "
                         "(e.g. 0.3 for short-clip demo).")
    ap.add_argument("--attacks", type=str, default=None,
                    help="Comma-separated list of attack names to run "
                         "(default: all). Available: " + ", ".join(ATTACK_REGISTRY))
    ap.add_argument("--success_threshold", type=float, default=0.9,
                    help="Bit accuracy threshold to count as a pass (default: 0.9).")
    ap.add_argument("--alpha_sweep", type=str, default=None,
                    help="Comma-separated alpha values to sweep over "
                         "(e.g. '0.005,0.01,0.02,0.04,0.08'). When set, the "
                         "script ignores --watermarked, re-embeds the --clean "
                         "audio at each alpha, runs the full attack battery, "
                         "and saves listening clips for each alpha.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    attacks = [a.strip() for a in args.attacks.split(",")] if args.attacks else None

    if args.alpha_sweep:
        if args.clean is None:
            raise SystemExit("--alpha_sweep requires --clean (the un-watermarked source).")
        alphas = [float(x) for x in args.alpha_sweep.split(",")]
        run_alpha_sweep(
            clean_path=args.clean,
            payload=args.payload,
            key=args.key,
            alphas=alphas,
            out_dir=args.out_dir,
            clip_seconds=args.clip_seconds,
            attacks=attacks,
            success_threshold=args.success_threshold,
        )
    else:
        if args.watermarked is None:
            raise SystemExit("--watermarked is required (or use --alpha_sweep).")
        run_evaluation(
            watermarked_path=args.watermarked,
            clean_path=args.clean,
            payload=args.payload,
            key=args.key,
            alpha=args.alpha,
            out_dir=args.out_dir,
            clip_seconds=args.clip_seconds,
            attacks=attacks,
            success_threshold=args.success_threshold,
        )


if __name__ == "__main__":
    main()