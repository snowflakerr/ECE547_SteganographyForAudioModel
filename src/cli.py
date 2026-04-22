"""
cmds:

synthesize —> TTS synthesis with embedded watermark
embed-music —> Embed watermark into an existing audio file
detect -> Detect and decode a watermark from a .wav file

usage:

python cli.py synthesize  --text "Hello world" --payload 42 --key "secret" --output speech.wav
python cli.py embed-music --input song.wav     --payload 42 --key "secret" --output song_wm.wav
python cli.py detect      --input speech.wav   --key "secret"
"""

import argparse

from watermark import FramewiseWatermarker
from music_pipe import MusicWatermarkPipeline
from tts_pipe   import WatermarkedTTSPipeline
from detector       import WatermarkDetector


#Argument parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="watermark",
        description="TraceableSpeech-style framewise audio watermarking toolkit.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    #synthesize
    s = sub.add_parser("synthesize", help="TTS synthesis with embedded watermark")
    s.add_argument("--text",    required=True,
                   help="Text to synthesize")
    s.add_argument("--payload", type=int, default=0,
                   help="Integer watermark payload, e.g. speaker / model ID (default 0)")
    s.add_argument("--output",  default="output.wav",
                   help="Output .wav file path (default: output.wav)")
    s.add_argument("--key",     default="secret_key",
                   help="Shared secret for PN carrier generation")
    s.add_argument("--alpha",   type=float, default=0.04,
                   help="Watermark embedding strength (default 0.04)")
    s.add_argument("--bits",    type=int, default=32,
                   help="Payload bit-width (default 32)")
    s.add_argument("--model",   default="tts_models/en/ljspeech/vits",
                   help="Coqui TTS model string (default: vits/ljspeech)")

    #embed-music
    m = sub.add_parser("embed-music", help="Embed watermark into an existing audio file")
    m.add_argument("--input",   required=True,
                   help="Input .wav file")
    m.add_argument("--output",  default="output_wm.wav",
                   help="Output .wav file path (default: output_wm.wav)")
    m.add_argument("--payload", type=int, default=0)
    m.add_argument("--key",     default="secret_key")
    m.add_argument("--alpha",   type=float, default=0.04)
    m.add_argument("--bits",    type=int, default=32)

    #detect
    d = sub.add_parser("detect", help="Detect and decode watermark from a .wav file")
    d.add_argument("--input",   required=True,
                   help="Input .wav file to inspect")
    d.add_argument("--key",     default="secret_key")
    d.add_argument("--bits",    type=int, default=32,
                   help="Payload bit-width (must match embedding config)")

    return p


#Command handlers

def cmd_synthesize(args) -> None:
    wm   = FramewiseWatermarker(
        secret_key=args.key, alpha=args.alpha, payload_bits=args.bits
    )
    pipe = WatermarkedTTSPipeline(wm, model_name=args.model)
    audio, sr = pipe.synthesize(
        args.text, payload=args.payload, output_path=args.output
    )
    print(f"[OK] {len(audio)/sr:.2f}s audio  →  {args.output}")


def cmd_embed_music(args) -> None:
    wm   = FramewiseWatermarker(
        secret_key=args.key, alpha=args.alpha, payload_bits=args.bits
    )
    pipe = MusicWatermarkPipeline(wm)
    pipe.embed_file(args.input, args.output, payload=args.payload)
    print(f"[OK] Watermarked audio saved  →  {args.output}")


def cmd_detect(args) -> None:
    wm  = FramewiseWatermarker(secret_key=args.key, payload_bits=args.bits)
    det = WatermarkDetector(wm)
    result = det.detect_file(args.input)
    det.print_result(result)

#Entry point

_HANDLERS = {
    "synthesize":  cmd_synthesize,
    "embed-music": cmd_embed_music,
    "detect":      cmd_detect,
}


def main() -> None:
    args = build_parser().parse_args()
    _HANDLERS[args.command](args)


if __name__ == "__main__":
    main()