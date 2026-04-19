import soundfile as sf
import librosa
from watermark_core import extract_energy_ratio_watermark

def attack_lossy_compression(input_wav, output_ogg):
    """
    Simulate a lossy compression attack by converting a high-fidelity WAV 
    into an OGG Vorbis file. This mimics MP3 behavior by discarding 
    high-frequency details and smoothing out minor magnitude variations.
    """
    print("--- Simulating Lossy Compression Attack (WAV -> OGG) ---")
    
    # 1. Load the watermarked high-fidelity WAV
    y, sr = librosa.load(input_wav, sr=None)
    
    # 2. Force export to OGG Vorbis format
    sf.write(output_ogg, y, sr, format='OGG', subtype='VORBIS')
    print(f"[+] Compression complete. High-fidelity WAV compressed to: {output_ogg}")
    
    return output_ogg

# ==========================================
# Red-Teaming Execution
# ==========================================

# 1. Subject the audio to compression degradation
attacked_audio = attack_lossy_compression("watermarked_by_me.wav", "attacked_audio.ogg")

# 2. Attempt to extract the watermark from the degraded audio
print("\n--- Starting Extraction from Compressed Audio ---")
# librosa natively supports reading .ogg formats for spectral analysis
extract_energy_ratio_watermark(attacked_audio)