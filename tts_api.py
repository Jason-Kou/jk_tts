#!/usr/bin/env python3
"""
Simple CLI wrapper for jk_tts - generates audio from text input.
Usage: uv run python tts_api.py --text "你好" --output /tmp/output.wav [--mode base] [--voice jason]
       uv run python tts_api.py --text-file /tmp/input.txt --output /tmp/output.wav
"""

import argparse
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.generate import generate_audio, load_audio
from mlx_audio.tts.utils import load_model

from cosyvoice3_backend import merge_wavs_with_ffmpeg, synthesize_segments as synthesize_cosyvoice3_segments
from moss_tts_backend import moss_model_for_mode, synthesize_segments as synthesize_moss_segments

VOICES_DIR = Path(__file__).parent / "voices"

MODELS = {
    "voice_design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "cosyvoice3": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "moss_local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
}

VOICE_PROFILES = {
    "jason": {
        "ref_audio": str(VOICES_DIR / "jason.wav"),
        "ref_text": "大家好,我是Jason.欢迎回到我的频道.今天给大家讲一段Tesla的故事",
    },
    "en_male": {   # ElevenLabs clone, ported from the CosyVoice pipeline
        "ref_audio": str(VOICES_DIR / "en_male.wav"),
        "ref_text": (
            "Need to send HDMI audio to a soundbar, receiver, or speakers? "
            "This 4K 60Hz HDMI audio extractor JTD-322 separates audio from your "
            "HDMI signal while still passing video to your TV, monitor, or projector."
        ),
        "lang": "english",
    },
    "official_female": {
        "ref_audio": str(VOICES_DIR / "official_female.wav"),
        "ref_text": "希望你以后能够做的比我还好呦。",
    },
}

DEFAULT_INSTRUCT = "A cheerful young female voice with clear pronunciation and moderate speed"
MAX_SEGMENT_CHARS = 200
MAX_TEXT_FILE_BYTES = 20_000


def split_text(text: str, max_segment_chars: int = MAX_SEGMENT_CHARS) -> list[str]:
    import re
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip().replace("\n", "") for p in paragraphs if p.strip()]
    
    segments = []
    buffer = ""
    for para in paragraphs:
        if len(para) <= max_segment_chars:
            if buffer and len(buffer) + len(para) > max_segment_chars:
                segments.append(buffer)
                buffer = para
            else:
                buffer = f"{buffer}{para}" if buffer else para
        else:
            if buffer:
                segments.append(buffer)
                buffer = ""
            # Split long paragraphs at sentence boundaries
            parts = re.split(r"(?<=[。！？；\?\!])", para)
            for sentence in [p.strip() for p in parts if p.strip()]:
                if buffer and len(buffer) + len(sentence) > max_segment_chars:
                    segments.append(buffer)
                    buffer = sentence
                else:
                    buffer = f"{buffer}{sentence}" if buffer else sentence
    if buffer:
        segments.append(buffer)
    return segments


def read_text_input(text: str | None, text_file: str | None) -> str:
    """Read bounded UTF-8 input without placing file contents in argv or logs."""
    if text is not None:
        value = text
    else:
        path = Path(text_file or "")
        if not path.exists() or path.is_symlink() or not path.is_file():
            raise ValueError("--text-file must be a regular, non-symbolic-link file")
        if path.stat().st_size > MAX_TEXT_FILE_BYTES:
            raise ValueError(f"--text-file exceeds {MAX_TEXT_FILE_BYTES} bytes")
        value = path.read_text(encoding="utf-8")
    value = value.strip()
    if not value:
        raise ValueError("text must not be empty")
    return value


def main():
    parser = argparse.ArgumentParser(description="jk_tts CLI")
    text_source = parser.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text", help="Text to synthesize")
    text_source.add_argument("--text-file", help="UTF-8 text file to synthesize (safer for services)")
    parser.add_argument("--output", required=True, help="Output audio file path (.wav or .mp3)")
    parser.add_argument("--mode", default="base", choices=["base", "voice_design", "cosyvoice3", "moss_local"])
    parser.add_argument("--voice", default="en_male", help="Voice profile name (for base/cosyvoice3/moss_* mode)")
    parser.add_argument(
        "--max-segment-chars",
        type=int,
        default=MAX_SEGMENT_CHARS,
        help="Maximum characters per synthesis segment (default: 200)",
    )
    args = parser.parse_args()

    if not 40 <= args.max_segment_chars <= MAX_SEGMENT_CHARS:
        parser.error(
            f"--max-segment-chars must be between 40 and {MAX_SEGMENT_CHARS}"
        )

    try:
        text = read_text_input(args.text, args.text_file)
    except (OSError, UnicodeError, ValueError) as error:
        parser.error(str(error))
    segments = split_text(text, args.max_segment_chars)
    print(f"Split into {len(segments)} segment(s)")

    if args.mode in ("cosyvoice3", "moss_local"):
        voice_profile = VOICE_PROFILES.get(args.voice)
        if voice_profile is None:
            print(f"Warning: voice '{args.voice}' not found, using default")
            voice_profile = VOICE_PROFILES["en_male"]

        with tempfile.TemporaryDirectory() as tmpdir:
            seg_files = [Path(tmpdir) / f"seg_{i:03d}.wav" for i in range(len(segments))]
            if args.mode == "cosyvoice3":
                synthesize_cosyvoice3_segments(segments, seg_files, voice_profile)
            else:
                synthesize_moss_segments(segments, seg_files, voice_profile, moss_model_for_mode(args.mode))
            merge_wavs_with_ffmpeg(seg_files, Path(args.output), volume=1.0)

        if Path(args.output).exists() and Path(args.output).stat().st_size > 0:
            print(f"SUCCESS: {args.output}")
            return
        print("ERROR: Output file not created")
        sys.exit(1)

    print(f"Loading model: {MODELS[args.mode]}...")
    model = load_model(MODELS[args.mode])
    print("Model loaded.")

    with tempfile.TemporaryDirectory() as tmpdir:
        seg_files = []
        for i, segment in enumerate(segments):
            print(f"  Generating segment {i+1}/{len(segments)}")
            seg_prefix = str(Path(tmpdir) / f"seg_{i:03d}")

            kwargs = {
                "model": model,
                "text": segment,
                "file_prefix": seg_prefix,
                "join_audio": True,
                "max_tokens": 4096,
                "lang_code": "chinese",
            }

            if args.mode == "voice_design":
                kwargs["instruct"] = DEFAULT_INSTRUCT
            elif args.mode == "base":
                if args.voice not in VOICE_PROFILES:
                    print(f"Warning: voice '{args.voice}' not found, using default")
                profile = dict(VOICE_PROFILES.get(args.voice, VOICE_PROFILES["en_male"]))
                # an English voice narrating in chinese mode picks up an accent
                kwargs["lang_code"] = profile.pop("lang", "chinese")
                kwargs.update(profile)

            generate_audio(**kwargs)
            seg_file = Path(f"{seg_prefix}.wav")
            if seg_file.exists():
                seg_files.append(seg_file)

        if not seg_files:
            print("ERROR: No audio generated")
            sys.exit(1)

        # Merge segments
        all_audio = [load_audio(str(f), sample_rate=model.sample_rate) for f in seg_files]
        merged = mx.concatenate(all_audio, axis=0)
        
        wav_path = str(Path(tmpdir) / "merged.wav")
        audio_write(wav_path, np.array(merged), model.sample_rate, format="wav")

        # Boost volume + convert to output format
        import subprocess
        output = args.output
        
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-filter:a", "volume=1.8",
            output
        ], capture_output=True)

        if Path(output).exists() and Path(output).stat().st_size > 0:
            print(f"SUCCESS: {output}")
        else:
            print("ERROR: Output file not created")
            sys.exit(1)


if __name__ == "__main__":
    main()
