from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

from audio_engine import OutputProcessor, SAMPLE_RATE, peak_normalize, resample, save_wav


ROOT = Path(__file__).resolve().parent
VOICE_DIR = ROOT / "coqui_project" / "voices"
DEFAULT_REFERENCE = ROOT / "coqui_project" / "reference.wav"
OUTPUT_DIR = ROOT.parent / "Generated_Outputs" / "User_Version"
UPLOAD_DIR = ROOT.parent / "Generated_Outputs" / "Uploaded_References" / "User_Version"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib_cache"))
os.environ.setdefault("VOICE_STUDIO_KEY", "voice_studio_private_key")


def available_references() -> dict[str, Path]:
    """Find saved voice reference WAVs that should appear in the UI."""
    voices = {}
    if DEFAULT_REFERENCE.exists():
        voices["Default Voice"] = DEFAULT_REFERENCE
    if VOICE_DIR.exists():
        for path in sorted(VOICE_DIR.glob("*.wav")):
            voices[path.stem.replace("_", " ").title()] = path
    return voices


@st.cache_resource(show_spinner=False)
def load_tts():
    """Load XTTS once per Streamlit session because model startup is expensive."""
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TTS(MODEL_NAME).to(device)


def synthesize(text: str, speaker_wav: Path, language: str) -> np.ndarray:
    """Generate speech with Coqui XTTS using a reference speaker WAV."""
    tts = load_tts()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    tts.tts_to_file(
        text=text,
        speaker_wav=str(speaker_wav),
        language=language,
        file_path=str(tmp_path),
        split_sentences=True,
    )

    audio, source_sr = sf.read(tmp_path)
    tmp_path.unlink(missing_ok=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return peak_normalize(resample(audio.astype(np.float32), int(source_sr), SAMPLE_RATE))


def output_id(text: str, voice_name: str, stamp: str) -> int:
    """Create a deterministic-looking 32-bit ID for this generated file."""
    digest = hashlib.sha256(f"{voice_name}:{stamp}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def main() -> None:
    st.set_page_config(page_title="Voice Studio", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #09090b; color: #f4f4f5; }
        .stButton button {
            border-radius: 8px;
            border: 1px solid #fafafa;
            background: #fafafa;
            color: #09090b;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Voice Studio")
    st.caption("Choose a voice, enter text, and generate audio.")

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        references = available_references()
        voice_source = st.radio("Voice source", ["Saved voice", "Upload voice"], horizontal=True)

        if voice_source == "Saved voice":
            if not references:
                st.error("No saved voices are available.")
                return
            voice_name = st.selectbox("Voice", list(references))
            speaker_wav = references[voice_name]
        else:
            uploaded = st.file_uploader("Voice sample", type=["wav"])
            voice_name = uploaded.name if uploaded else "Uploaded Voice"
            if uploaded:
                upload_dir = UPLOAD_DIR
                upload_dir.mkdir(parents=True, exist_ok=True)
                speaker_wav = upload_dir / uploaded.name
                speaker_wav.write_bytes(uploaded.getbuffer())
            else:
                speaker_wav = None

        language = st.selectbox(
            "Language",
            ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"],
            index=0,
        )

        text = st.text_area(
            "What should the voice say?",
            value="Hello, this is a generated voice sample.",
            height=170,
        )
        generate = st.button("Generate Audio", type="primary", use_container_width=True)

    with right:
        if not generate:
            st.info("Your generated audio will appear here.")
            return

        if not text.strip():
            st.error("Enter text to generate audio.")
            return
        if speaker_wav is None:
            st.error("Upload a voice sample first.")
            return

        try:
            with st.spinner("Generating audio..."):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                generated = synthesize(text, speaker_wav, language)
                item_id = output_id(text, voice_name, stamp)

                # Finalize the audio before saving. The user-facing UI does
                # not disclose this step, but the checker can recover item_id.
                processor = OutputProcessor(os.environ["VOICE_STUDIO_KEY"])
                final_audio = processor.process(generated, item_id, SAMPLE_RATE)
                output_path = OUTPUT_DIR / f"generated_audio_{stamp}.wav"
                audio_bytes = save_wav(output_path, final_audio)

            st.audio(audio_bytes, format="audio/wav")
            st.download_button("Download Audio", audio_bytes, output_path.name, "audio/wav")
            st.success("Audio generated.")
        except ModuleNotFoundError as exc:
            st.error(f"Missing dependency: {exc.name}. Install dependencies with pip install -r requirements.txt.")
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
