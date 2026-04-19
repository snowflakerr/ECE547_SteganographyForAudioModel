import librosa
import soundfile as sf
from watermark_core import extract_energy_ratio_watermark

def attack_time_stretch(input_wav, output_wav, rate=1.2):
    """
    Simulate a Time-Scale Modification (TSM) attack.
    rate > 1.0 means speed-up (audio becomes shorter, fewer frames).
    rate < 1.0 means slow-down (audio becomes longer, more frames).
    """
    print(f"--- Simulating TSM Speed Attack (Rate: {rate}x) ---")
    y, sr = librosa.load(input_wav, sr=None)
    
    # Forcibly alter the audio duration (uses a Phase Vocoder under the hood).
    # This step completely disrupts the original STFT time-frame alignment!
    y_stretched = librosa.effects.time_stretch(y, rate=rate)
    
    sf.write(output_wav, y_stretched, sr)
    print(f"[+] Attack complete. Time-stretched audio saved to: {output_wav}")
    return output_wav

# ==========================================
# Red-Teaming Execution
# ==========================================

# 1. Apply a 1.2x speed-up attack.
# You can also change the rate to 0.8 to test a slow-down attack.
attacked_audio = attack_time_stretch("watermarked_by_me.wav", "attacked_speedup.wav", rate=1.2)

# 2. Attempt to extract the watermark from the time-distorted audio.
print("\n--- Starting Extraction from Time-Stretched Audio ---")
extract_energy_ratio_watermark(attacked_audio)