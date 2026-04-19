import numpy as np
import librosa
import soundfile as sf
import scipy.signal
from watermark_core import extract_energy_ratio_watermark

def attack_highpass_filter(input_wav, output_wav, cutoff_freq=1500, order=5):
    """
    Simulate a High-Pass Filter (HPF) attack using a Butterworth filter.
    This will aggressively attenuate frequencies below the cutoff_freq.
    
    cutoff_freq: The frequency boundary in Hz. Frequencies below this are silenced.
    order: The steepness of the filter cut. Higher order = sharper cut.
    """
    print(f"--- Simulating High-Pass Filter Attack (Cutoff: {cutoff_freq} Hz) ---")
    
    # 1. Load the watermarked audio
    y, sr = librosa.load(input_wav, sr=None)
    
    # 2. Calculate the Nyquist frequency (half the sample rate)
    nyquist = 0.5 * sr
    
    # 3. Normalize the cutoff frequency for the SciPy filter
    normalized_cutoff = cutoff_freq / nyquist
    
    # 4. Design the Butterworth High-Pass Filter
    b, a = scipy.signal.butter(N=order, Wn=normalized_cutoff, btype='highpass')
    
    # 5. Apply the filter to the waveform using filtfilt (zero-phase filtering)
    y_filtered = scipy.signal.filtfilt(b, a, y)
    
    sf.write(output_wav, y_filtered, sr)
    print(f"[+] Attack complete. Filtered audio saved to: {output_wav}")
    return output_wav

# ==========================================
# Red-Teaming Execution: The Frequency Scalpel
# ==========================================

# Apply a 1500 Hz High-Pass Filter 
# This cuts directly into Band A (approx. 1170 - 1750 Hz), heavily testing the algorithm
attacked_audio = attack_highpass_filter("watermarked_by_me.wav", "attacked_hpf.wav", cutoff_freq=1500)

print("\n--- Starting Extraction from Filtered Audio ---")
extract_energy_ratio_watermark(attacked_audio)