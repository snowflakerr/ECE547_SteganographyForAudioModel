import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def compare_spectrograms(original_path, watermarked_path, output_image_name):
    print(f"Loading files: {original_path} & {watermarked_path}...")
    
    # 1. Load audio files (preserve original sample rate)
    y_orig, sr = librosa.load(original_path, sr=None)
    y_wm, _ = librosa.load(watermarked_path, sr=sr)
    
    # Ensure identical lengths to avoid broadcasting errors
    min_len = min(len(y_orig), len(y_wm))
    y_orig = y_orig[:min_len]
    y_wm = y_wm[:min_len]

    # 2. Compute STFT and convert to decibels (dB)
    D_orig = librosa.amplitude_to_db(np.abs(librosa.stft(y_orig)), ref=np.max)
    D_wm = librosa.amplitude_to_db(np.abs(librosa.stft(y_wm)), ref=np.max)

    # 3. Calculate the residual (Absolute difference)
    # This reveals exactly which frequencies the AI modified.
    D_diff = np.abs(D_wm - D_orig)

    # 4. Plotting configuration
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Original Audio
    librosa.display.specshow(D_orig, sr=sr, x_axis='time', y_axis='hz', ax=ax[0], cmap='viridis')
    ax[0].set_title('Original Speech Spectrogram', fontsize=12)
    fig.colorbar(ax[0].collections[0], ax=ax[0], format="%+2.0f dB")

    # Plot 2: Watermarked Audio
    librosa.display.specshow(D_wm, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='viridis')
    ax[1].set_title("Watermarked Speech (AI Model)", fontsize=12)
    fig.colorbar(ax[1].collections[0], ax=ax[1], format="%+2.0f dB")

    # Plot 3: Difference/Residual
    img_diff = librosa.display.specshow(D_diff, sr=sr, x_axis='time', y_axis='hz', ax=ax[2], cmap='magma')
    ax[2].set_title('Absolute Difference (Watermark Injection Pattern)', fontsize=12, color='red')
    fig.colorbar(img_diff, ax=ax[2], format="%+2.0f dB")

    plt.tight_layout()
    plt.savefig(output_image_name, dpi=300)
    print(f"Saved visualization to {output_image_name}\n")
    
    # Optional: comment out the next line if you don't want the UI window to pop up
    plt.show()

# Process first pair: Hello World
compare_spectrograms(
    "hello_world_clean.wav", 
    "hello_world_watermarked.wav", 
    "hello_world_analysis.png"
)

# Process second pair: Scotty 5s
compare_spectrograms(
    "scotty_5s_clean.wav", 
    "scotty_5s_watermarked.wav", 
    "scotty_5s_analysis.png"
)