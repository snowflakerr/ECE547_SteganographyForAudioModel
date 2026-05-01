from pathlib import Path
import numpy as np

from generate_demo import FramewiseWatermarker, load_and_trim, apply_watermark, save, snr_db

out = Path("../aaron_attack_results")
out.mkdir(parents=True, exist_ok=True)

# Try both possible locations
candidates = [Path("../scotty.wav"), Path("./scotty.wav")]
scotty_path = next((p for p in candidates if p.exists()), None)

if scotty_path is None:
    raise FileNotFoundError("Could not find scotty.wav in repo root or src folder.")

payload = 42
key = "traceable_speech_demo"
alpha = 0.01
sample_rate = 44100

wm = FramewiseWatermarker(
    secret_key=key,
    alpha=alpha,
    payload_bits=32,
    perceptual=True,
)

print("Loading Scotty audio...")
scotty = load_and_trim(str(scotty_path), duration_s=5.0, sample_rate=sample_rate)

peak = np.abs(scotty).max()
if peak > 0:
    scotty = scotty / peak * 0.9

print("Saving clean + watermarked Scotty files...")
save(str(out / "scotty_5s_clean.wav"), scotty, sample_rate)

scotty_wm = apply_watermark(scotty, sample_rate, wm, payload)
save(str(out / "scotty_5s_watermarked.wav"), scotty_wm, sample_rate)

print(f"SNR: {snr_db(scotty, scotty_wm):.1f} dB")
print("DONE. Use aaron_attack_results/scotty_5s_watermarked.wav for your attack tests.")
