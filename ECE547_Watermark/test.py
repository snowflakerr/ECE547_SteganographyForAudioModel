from watermark_core import embed_energy_ratio_watermark, extract_energy_ratio_watermark

print("--- Starting Watermark Pipeline ---")

# 1. Embed the watermark into the raw Coqui TTS output
# Ensure 'anna_tts_output.wav' is in the same directory as this script
embed_energy_ratio_watermark(
    input_wav="anna_tts_output.wav", 
    output_wav="watermarked_by_me.wav", 
    alpha=0.2
)

# 2. Extract and verify the watermark from the newly generated file
print("\n--- Starting Verification Phase ---")
extract_energy_ratio_watermark("watermarked_by_me.wav")