import numpy as np
import librosa
import soundfile as sf
from watermark_core import extract_energy_ratio_watermark

def attack_awgn(input_wav, output_wav, snr_db=20):
    """
    Simulate Additive White Gaussian Noise (AWGN) attack.
    snr_db: Signal-to-Noise Ratio in decibels. 
            Lower SNR means louder noise. 20dB is a standard audible noise level.
    """
    print(f"--- Simulating AWGN Attack (SNR: {snr_db}dB) ---")
    y, sr = librosa.load(input_wav, sr=None)
    
    # 1. Calculate the Root Mean Square (RMS) of the original signal
    rms_signal = np.sqrt(np.mean(y**2))
    
    # 2. Calculate the required RMS of the noise based on the target SNR
    # Formula: SNR = 20 * log10(RMS_signal / RMS_noise)
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    
    # 3. Generate standard white noise and scale it to the target RMS
    noise = np.random.normal(0, rms_noise, y.shape)
    
    # 4. Add the noise directly to the time-domain waveform
    y_noisy = y + noise
    
    sf.write(output_wav, y_noisy, sr)
    print(f"[+] Attack complete. Noisy audio saved to: {output_wav}")
    return output_wav

# ==========================================
# Red-Teaming Execution
# ==========================================

# Apply a 20dB AWGN attack (Medium-high noise level)
attacked_audio = attack_awgn("watermarked_by_me.wav", "attacked_noise.wav", snr_db=20)

# Attempt to recover the watermark from the noisy file
print("\n--- Starting Extraction from Noisy Audio ---")
extract_energy_ratio_watermark(attacked_audio)