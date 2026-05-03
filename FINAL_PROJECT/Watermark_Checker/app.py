from __future__ import annotations

import pandas as pd
import soundfile as sf
import streamlit as st

from watermark_codec import decode_payload


def main() -> None:
    st.set_page_config(page_title="Watermark Checker", layout="wide")
    st.title("Watermark Checker")
    st.caption("Upload generated audio and decode the embedded payload.")

    left, right = st.columns([0.42, 0.58], gap="large")

    with left:
        uploaded = st.file_uploader("Audio file", type=["wav"])
        secret_key = st.text_input("Secret key", value="voice_studio_private_key")
        bits = st.number_input("Payload bits", min_value=8, max_value=64, value=32, step=1)
        check = st.button("Check Audio", type="primary", use_container_width=True)

    with right:
        if not check:
            st.info("Decoded payload details will appear here.")
            return
        if uploaded is None:
            st.error("Upload a WAV file first.")
            return

        try:
            audio, sample_rate = sf.read(uploaded)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)

            result = decode_payload(audio, int(sample_rate), secret_key, int(bits))
            st.caption(f"Uploaded file: {uploaded.name}")
            st.caption("Decoder: Nick STFT spread-spectrum")

            st.metric("Decoded Payload ID", str(result.payload_id))
            st.metric("Mean Bit Confidence", f"{result.mean_confidence:.3f}")
            st.code(result.bit_string, language="text")

            if result.mean_confidence < 0.55:
                st.info(
                    "Low-confidence decodes can still produce a numeric ID. "
                    "Use the secret key and payload-bit settings that were used "
                    "when embedding the watermark."
                )

            bit_table = pd.DataFrame(
                {
                    "bit": list(range(int(bits))),
                    "score": result.bit_scores,
                    "confidence": result.bit_confidence,
                    "value": [1 if score > 0 else 0 for score in result.bit_scores],
                }
            )
            st.subheader("Bit Details")
            st.dataframe(bit_table, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
