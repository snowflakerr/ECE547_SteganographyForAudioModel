from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import soundfile as sf
import streamlit as st

from watermark_codec import decode_payload


ROOT = Path(__file__).resolve().parent
USER_RECORDS = ROOT.parent / "Generated_Outputs" / "User_Version" / ".generation_records.jsonl"
DEVELOPER_RECORDS = ROOT.parent / "Generated_Outputs" / "Developer_Version" / ".generation_records.jsonl"
DEFAULT_RECORDS = USER_RECORDS
SUPPORTED_ENCODER_VERSION = "fsk-tail-v2"


def load_records(path: Path) -> pd.DataFrame:
    """Load JSONL generation records written by either generator app."""
    if not path.exists():
        return pd.DataFrame()
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def matching_filename(records: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Return record rows that exactly match the uploaded filename."""
    if records.empty or "file" not in records.columns:
        return pd.DataFrame()
    return records[records["file"] == filename]


def source_hint(filename: str) -> str | None:
    """Warn when the selected records source probably does not match the file."""
    if filename.startswith("voice_watermarked_"):
        return (
            "This filename looks like a Developer_Version output. Select "
            "Developer Version under generation records before checking it."
        )
    if filename.startswith("voice_clean_"):
        return (
            "This filename looks like a clean Developer_Version output, so it is "
            "not expected to match User_Version generation records."
        )
    if not filename.startswith("generated_audio_"):
        return (
            "This filename does not look like the current User_Version output "
            "naming pattern: generated_audio_*.wav."
        )
    return None


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

        record_source = st.radio("Generation records", ["User Version", "Developer Version", "Custom"], horizontal=True)
        default_path = {
            "User Version": USER_RECORDS,
            "Developer Version": DEVELOPER_RECORDS,
            "Custom": DEFAULT_RECORDS,
        }[record_source]
        records_path = st.text_input("Records path", value=str(default_path))

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
            hint = source_hint(uploaded.name)
            if hint:
                st.warning(hint)

            st.metric("Decoded Payload ID", str(result.payload_id))
            st.metric("Mean Bit Confidence", f"{result.mean_confidence:.3f}")
            st.code(result.bit_string, language="text")

            records = load_records(Path(records_path))
            filename_matches = matching_filename(records, uploaded.name)
            if not records.empty and "output_id" in records.columns:
                # A numeric decode is only meaningful when it matches a row in
                # the generation records created by the generator app.
                matches = records[records["output_id"] == result.payload_id]
                if not matches.empty:
                    st.success("Payload matched a known generation record.")
                    st.dataframe(matches, use_container_width=True, hide_index=True)
                else:
                    if not filename_matches.empty:
                        expected = filename_matches.iloc[0]["output_id"]
                        version = filename_matches.iloc[0].get("encoder_version")
                        if not version:
                            st.warning(
                                "This file is in the records, but it was generated before "
                                "the checker-compatible encoder was added. Regenerate the "
                                "audio in the current User_Version and check that new file."
                            )
                        elif version != SUPPORTED_ENCODER_VERSION:
                            st.warning(
                                f"This file uses encoder version {version}, but this checker "
                                f"expects {SUPPORTED_ENCODER_VERSION}."
                            )
                        else:
                            st.warning(
                                "This filename exists in the generation records, but the decoded "
                                f"payload does not match it. Expected {expected}; decoded "
                                f"{result.payload_id}."
                            )
                        st.dataframe(filename_matches, use_container_width=True, hide_index=True)
                    elif result.mean_confidence < 0.55:
                        st.warning(
                            "No matching generation record was found, and confidence is low. "
                            "Treat this as inconclusive rather than a positive detection."
                        )
                    else:
                        st.warning("No matching generation record was found.")

            if result.mean_confidence < 0.55:
                st.info(
                    "Low-confidence decodes can still produce a numeric ID. The useful signal "
                    "is whether that ID matches a known generation record."
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
