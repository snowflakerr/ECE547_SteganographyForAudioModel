import numpy as np
import librosa
import soundfile as sf

def embed_energy_ratio_watermark(input_wav, output_wav, alpha=0.2):
    """
    Embed a watermark using the Energy Ratio (Patchwork) method.
    This method provides redundancy across frames, making it highly robust 
    against Time-Scale Modification (slowdowns/speedups).
    
    alpha: The embedding strength. Higher = more robust but potentially audible. (0.1 ~ 0.3 recommended)
    """
    # Load the raw TTS audio (preserve original sample rate)
    y, sr = librosa.load(input_wav, sr=None)
    
    # 1. Apply STFT to convert the waveform into the frequency domain
    D = librosa.stft(y, n_fft=2048)
    magnitude = np.abs(D)
    phase = np.angle(D)
    
    # 2. Define adjacent frequency sub-bands in the mid-frequency range
    # Band A: bins 100-150, Band B: bins 151-200
    band_A_range = (100, 150)
    band_B_range = (150, 200)
    
# 3. Embedding Logic (Redundant Tiling with Swap Fix)
    for t in range(magnitude.shape[1]):
        energy_A = np.mean(magnitude[band_A_range[0]:band_A_range[1], t])
        energy_B = np.mean(magnitude[band_B_range[0]:band_B_range[1], t])
        
        # [NEW FIX] Guarantee Energy(A) > Energy(B) before scaling
        if energy_A < energy_B:
            # Swap the magnitude arrays for this time frame
            temp = np.copy(magnitude[band_A_range[0]:band_A_range[1], t])
            magnitude[band_A_range[0]:band_A_range[1], t] = magnitude[band_B_range[0]:band_B_range[1], t]
            magnitude[band_B_range[0]:band_B_range[1], t] = temp
        
        # Now apply the alpha scaling to deepen the gap and make it robust
        magnitude[band_A_range[0]:band_A_range[1], t] *= (1 + alpha)
        magnitude[band_B_range[0]:band_B_range[1], t] *= (1 - alpha)

    # 4. Energy Conservation and Signal Reconstruction
    # Reconstruct the complex STFT matrix using the modified magnitude and the original phase
    D_modified = magnitude * np.exp(1j * phase)
    
    # 5. Apply Inverse STFT (ISTFT) to convert back to the time domain
    y_watermarked = librosa.istft(D_modified)
    
    # Save the protected audio
    sf.write(output_wav, y_watermarked, sr)
    print(f"[+] Watermark successfully embedded. Saved to: {output_wav}")

def extract_energy_ratio_watermark(test_wav):
    """
    Blind extraction of the watermark by evaluating the energy ratio 
    between the predefined sub-bands.
    """
    y, sr = librosa.load(test_wav, sr=None)
    
    # Convert test audio to frequency domain
    D = librosa.stft(y, n_fft=2048)
    magnitude = np.abs(D)
    
    band_A_range = (100, 150)
    band_B_range = (150, 200)
    
    success_count = 0
    total_frames = magnitude.shape[1]
    
    # Check the energy ratio for every single time frame
    for t in range(total_frames):
        energy_A = np.mean(magnitude[band_A_range[0]:band_A_range[1], t])
        energy_B = np.mean(magnitude[band_B_range[0]:band_B_range[1], t])
        
        # If the rule Energy(A) > Energy(B) holds true, count it as a successful bit extraction
        if energy_A > energy_B:
            success_count += 1
            
    # Calculate the overall detection rate across the entire audio file
    detection_rate = success_count / total_frames
    print(f"[*] Watermark Detection Rate (Provenance Confidence): {detection_rate:.2%}")
    
    # We use an 80% threshold to account for frames broken by translation/time-scaling
    if detection_rate > 0.8: 
        print("[+] Verification SUCCESS: Authentic watermarked audio detected.")
        return True
    else:
        print("[-] Verification FAILED: Missing or destroyed watermark.")
        return False