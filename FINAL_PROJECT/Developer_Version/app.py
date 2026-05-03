from __future__ import annotations

import hashlib
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import soundfile as sf
import streamlit as st

from audio_engine import ENCODER_VERSION, OutputProcessor, SAMPLE_RATE, save_wav

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"


OUT_DIR = ROOT.parent / "Generated_Outputs" / "Developer_Version"
UPLOAD_DIR = ROOT.parent / "Generated_Outputs" / "Uploaded_References" / "Developer_Version"
VOICE_DIR = ROOT / "coqui_project" / "voices"
DEFAULT_REFERENCE = ROOT / "coqui_project" / "reference.wav"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

os.environ.setdefault("VOICE_STUDIO_KEY", "voice_studio_private_key")


def _peak_normalize(audio: np.ndarray, target: float = 0.9) -> np.ndarray:
    """Normalize Coqui output so strength settings behave consistently."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * target
    return audio.astype(np.float32)


def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    """Resample Coqui output to the sample rate used by the encoder/checker."""
    if source_sr == target_sr:
        return audio.astype(np.float32)
    from math import gcd
    from scipy.signal import resample_poly

    factor = gcd(source_sr, target_sr)
    return resample_poly(audio, target_sr // factor, source_sr // factor).astype(np.float32)


def available_references() -> dict[str, Path]:
    """Find saved Coqui speaker references for the developer UI."""
    voices = {}
    if DEFAULT_REFERENCE.exists():
        voices["Default reference"] = DEFAULT_REFERENCE
    if VOICE_DIR.exists():
        for path in sorted(VOICE_DIR.glob("*.wav")):
            voices[path.stem.replace("_", " ").title()] = path
    return voices


@st.cache_resource(show_spinner=False)
def load_coqui():
    """Load XTTS once per Streamlit session because the model is large."""
    import torch
    from TTS.api import TTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return TTS(XTTS_MODEL).to(device)


def synthesize_xtts(text: str, speaker_wav: Path | list[Path], language: str) -> np.ndarray:
    """Generate speech from text and a Coqui XTTS reference voice."""
    tts = load_coqui()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    speaker_arg = [str(path) for path in speaker_wav] if isinstance(speaker_wav, list) else str(speaker_wav)
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_arg,
        language=language,
        file_path=str(tmp_path),
        split_sentences=True,
    )

    audio, source_sr = sf.read(tmp_path)
    tmp_path.unlink(missing_ok=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return _peak_normalize(_resample(audio, int(source_sr), SAMPLE_RATE))


def make_payload(text: str, voice_label: str) -> int:
    """Derive a repeatable automatic payload from the selected voice and text."""
    digest = hashlib.sha256(f"{voice_label}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def waveform_plot(clean: np.ndarray, watermarked: np.ndarray) -> go.Figure:
    """Build a small before/after waveform plot for the developer demo."""
    max_points = 2_000
    step = max(1, len(clean) // max_points)
    t = np.arange(0, len(clean), step) / SAMPLE_RATE

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=clean[::step], name="Clean", line=dict(width=1)))
    fig.add_trace(go.Scatter(x=t, y=watermarked[::step], name="Watermarked", line=dict(width=1)))
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f4f4f5"),
        xaxis_title="Seconds",
        yaxis_title="Amplitude",
        legend=dict(orientation="h", y=1.1),
    )
    fig.update_xaxes(gridcolor="#27272a")
    fig.update_yaxes(gridcolor="#27272a")
    return fig


def main() -> None:
    st.set_page_config(page_title="Voice Watermark Studio", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: #09090b; color: #f4f4f5; }
        div[data-testid="stMetric"] {
            background: #18181b;
            border: 1px solid #27272a;
            padding: 14px 16px;
            border-radius: 8px;
        }
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

    st.title("Voice Watermark Studio")
    st.caption("Choose a Coqui XTTS voice reference, type speech, and generate watermarked audio.")

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        references = available_references()
        voice_source = st.radio("Voice source", ["Saved reference", "Upload reference"], horizontal=True)

        uploaded_path = None
        if voice_source == "Saved reference":
            if not references:
                st.error("Add at least one WAV reference to coqui_project/voices or coqui_project/reference.wav.")
                return
            voice_label = st.selectbox("Voice reference", list(references))
            speaker_wav = references[voice_label]
        else:
            uploaded = st.file_uploader("Reference WAV", type=["wav"])
            voice_label = uploaded.name if uploaded else "Uploaded reference"
            if uploaded:
                upload_dir = UPLOAD_DIR
                upload_dir.mkdir(parents=True, exist_ok=True)
                uploaded_path = upload_dir / uploaded.name
                uploaded_path.write_bytes(uploaded.getbuffer())
                speaker_wav = uploaded_path
            else:
                speaker_wav = None

        language = st.selectbox("Language", ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"], index=0)

        text = st.text_area(
            "What should the voice say?",
            value="Hello, this is a watermarked voice sample.",
            height=150,
        )

        with st.expander("Watermark settings"):
            payload_mode = st.radio("Payload", ["Automatic", "Manual"], horizontal=True)
            if payload_mode == "Manual":
                manual_payload = st.number_input("Payload ID", min_value=0, max_value=2**32 - 1, value=1)
            else:
                manual_payload = None
                st.caption("Automatic mode derives a payload ID from the selected voice and text.")
            secret_key = st.text_input("Secret key", value=os.environ["VOICE_STUDIO_KEY"])
            alpha = st.slider("Strength", 0.005, 0.08, 0.03, 0.001, format="%.3f")

        generate = st.button("Generate Watermarked Audio", type="primary", use_container_width=True)

    with right:
        if generate:
            if not text.strip():
                st.error("Enter text to synthesize.")
                return
            if speaker_wav is None:
                st.error("Upload a WAV reference voice first.")
                return

            try:
                with st.spinner("Synthesizing with Coqui XTTS and watermarking audio..."):
                    clean = synthesize_xtts(text, speaker_wav, language)

                    payload = int(manual_payload) if payload_mode == "Manual" else make_payload(text, voice_label)

                    # Developer_Version is transparent about this step. It uses
                    # the same STFT spread-spectrum encoder as User_Version so
                    # one checker can verify outputs from either app.
                    processor = OutputProcessor(secret_key=secret_key, strength=alpha, bits=32)
                    watermarked = processor.process(clean, payload, SAMPLE_RATE)

                    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    clean_path = OUT_DIR / f"voice_clean_{stamp}.wav"
                    wm_path = OUT_DIR / f"voice_watermarked_{stamp}.wav"
                    clean_bytes = save_wav(clean_path, clean)
                    wm_bytes = save_wav(wm_path, watermarked)

                metric_cols = st.columns(3)
                metric_cols[0].metric("Payload", str(payload))
                metric_cols[1].metric("Encoder", ENCODER_VERSION)
                metric_cols[2].metric("Duration", f"{len(watermarked) / SAMPLE_RATE:.2f}s")

                st.audio(wm_bytes, format="audio/wav")
                st.plotly_chart(waveform_plot(clean, watermarked), use_container_width=True)

                dl_cols = st.columns(2)
                dl_cols[0].download_button("Download clean WAV", clean_bytes, clean_path.name, "audio/wav")
                dl_cols[1].download_button("Download watermarked WAV", wm_bytes, wm_path.name, "audio/wav")

                st.success(f"Saved to {wm_path}")
            except ModuleNotFoundError as exc:
                st.error(f"Missing required dependency: {exc.name}. Install dependencies with: pip install -r requirements.txt")
            except Exception as exc:
                st.error(str(exc))
        else:
            st.info("Generated audio will appear here.")


if __name__ == "__main__":
    main()
