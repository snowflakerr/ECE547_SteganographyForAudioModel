# TeamSteganography Final Project

This final project contains a Coqui XTTS voice-generation demo, a developer
watermarking demo, a watermark checker, and Aaron's attack recordings.

## Folder Layout

```text
FINAL_PROJECT/
  User_Version/          User-facing voice generator
  Developer_Version/     Transparent developer/demo version
  Watermark_Checker/     Developer tool for decoding payload IDs
  Attack_Outputs/        Aaron's recorded attack samples
  Generated_Outputs/     Shared generated audio and records
```

`Generated_Outputs/` is intentionally shared by both generators so the checker
can read one predictable place for records.

## Setup

Use one shared virtual environment from this folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first Coqui XTTS run may download model files.

## Run The Apps

User-facing generator:

```bash
source .venv/bin/activate
cd FINAL_PROJECT/User_Version
streamlit run app.py
```

Developer generator:

```bash
source .venv/bin/activate
cd FINAL_PROJECT/Developer_Version
streamlit run app.py
```

Watermark checker:

```bash
source .venv/bin/activate
cd FINAL_PROJECT/Watermark_Checker
streamlit run app.py
```

## Demo Flow

1. Generate audio in `User_Version` or `Developer_Version`.
2. Open `Watermark_Checker`.
3. Upload the generated `.wav`.
4. Choose the matching generation records source.
5. Confirm the decoded payload matches a known record.

Aaron's attack files are in `FINAL_PROJECT/Attack_Outputs/` and can be uploaded
directly into `Watermark_Checker` during the demo.

## Notes

- Do not copy virtual environment folders into the final subfolders.
- Generated WAVs and record files are written under `FINAL_PROJECT/Generated_Outputs/`.
- The user-facing app intentionally does not mention watermarking.
