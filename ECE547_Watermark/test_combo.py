import numpy as np
import librosa
import soundfile as sf
from watermark_core import extract_energy_ratio_watermark

def attack_combo(input_wav, output_ogg, rate=1.2, snr_db=20):
    """
    Simulate a realistic worst-case scenario by chaining 3 attacks:
    1. Time-Scale Modification (Speed up)
    2. Additive White Gaussian Noise (AWGN)
    3. Lossy Compression (Save as OGG)
    """
    print(f"--- Simulating Ultimate Combo Attack (TSM {rate}x + AWGN {snr_db}dB + OGG Compression) ---")
    
    # Load the watermarked audio
    y, sr = librosa.load(input_wav, sr=None)
    
    # Strike 1: Time-Scale Modification (TSM)
    print("[*] Strike 1: Applying Phase Vocoder time-stretch...")
    y_combo = librosa.effects.time_stretch(y, rate=rate)
    
    # Strike 2: Additive White Gaussian Noise (AWGN)
    print("[*] Strike 2: Injecting background white noise...")
    rms_signal = np.sqrt(np.mean(y_combo**2))
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.normal(0, rms_noise, y_combo.shape)
    y_combo = y_combo + noise
    
    # Strike 3: Lossy Compression
    # Saving the heavily degraded waveform directly into a compressed format
    print("[*] Strike 3: Forcing lossy OGG compression...")
    sf.write(output_ogg, y_combo, sr, format='OGG', subtype='VORBIS')
    
    print(f"[+] Combo attack complete! Devastated audio saved to: {output_ogg}")
    return output_ogg

# ==========================================
# Red-Teaming Execution (The Ultimate Boss)
# ==========================================

attacked_audio = attack_combo("watermarked_by_me.wav", "attacked_combo.ogg", rate=1.2, snr_db=20)

print("\n--- Starting Extraction from Combo Attacked Audio ---")
extract_energy_ratio_watermark(attacked_audio)