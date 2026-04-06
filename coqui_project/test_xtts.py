from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

tts.tts_to_file(
    text="Hello, this is a test of Coqui XTTS on macOS.",
    speaker_wav="reference.wav",
    language="en",
    file_path="output.wav",
)
