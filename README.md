# Audio Watermarking for Voice Generation

## What this project does
This project explores how to watermark the **audio output** of an open-source voice generation model for a security engineering class.

The idea is:
- generate speech with an open-source TTS model
- embed a watermark into the generated audio
- test whether the watermark can still be detected after edits like compression, noise, trimming, or resampling

We are focusing on **output watermarking**, not watermarking the model weights.

## Main tools
- **Coqui TTS / XTTS v2** for voice generation
- **Custom watermarking pipeline** for watermark embedding and detection

## Dependencies
- Python 3.11
- coqui-tts
- torch
- torchaudio
- torchcodec
- transformers
- ffmpeg
- git

## Main links
- **Coqui TTS GitHub:** https://github.com/coqui-ai/TTS
- **Coqui TTS docs:** https://coqui-tts.readthedocs.io/
- **PyTorch:** https://pytorch.org/
- **FFmpeg:** https://ffmpeg.org/

## Reference papers
- [1] F. Kreuk, Y. Adi, B. Raj, R. Singh, and J. Keshet, “Hide and Speak: Towards Deep Neural Networks for Speech Steganography,” arXiv preprint arXiv:1902.03083, 2020. [Online]. Available: https://arxiv.org/abs/1902.03083
- [2] Y. Wen, A. Innuganti, A. B. Ramos, H. Guo, and Q. Yan, “SoK: How Robust is Audio Watermarking in Generative AI Models?,” arXiv preprint arXiv:2503.19176, 2025. [Online]. Available: https://arxiv.org/abs/2503.19176
- [3] Y. Wang, Z. Chen, X. Zhang, and H. Li, “TraceableSpeech: Towards Proactively Traceable Text-to-Speech with Watermarking,” arXiv preprint arXiv:2406.04840, 2024. [Online]. Available: https://arxiv.org/abs/2406.04840

## Current goal
Build a working pipeline that:
1. generates speech
2. watermarks the generated audio
3. tests whether the watermark survives realistic modifications

## Notes
This repository does **not** track the Python virtual environment. Use the dependency list above to recreate the environment locally.
