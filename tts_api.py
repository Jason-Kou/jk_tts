#!/usr/bin/env python3
"""
Simple CLI wrapper for jk_tts - generates audio from text input.
Usage: uv run python tts_api.py --text "你好" --output /tmp/output.wav [--mode base] [--voice jason]
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
    "official_female": {
        "ref_audio": str(VOICES_DIR / "official_female.wav"),
        "ref_text": "希望你以后能够做的比我还好呦。",
    },
}

DEFAULT_INSTRUCT = "A cheerful young female voice with clear pronunciation and moderate speed"
MAX_SEGMENT_CHARS = 200


def split_text(text: str) -> list[str]:
    import re
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip().replace("\n", "") for p in paragraphs if p.strip()]
    
    segments = []
    buffer = ""
    for para in paragraphs:
        if len(para) <= MAX_SEGMENT_CHARS:
            if buffer and len(buffer) + len(para) > MAX_SEGMENT_CHARS:
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
                if buffer and len(buffer) + len(sentence) > MAX_SEGMENT_CHARS:
                    segments.append(buffer)
                    buffer = sentence
                else:
                    buffer = f"{buffer}{sentence}" if buffer else sentence
    if buffer:
        segments.append(buffer)
    return segments


def main():
    parser = argparse.ArgumentParser(description="jk_tts CLI")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio file path (.wav or .mp3)")
    parser.add_argument("--mode", default="base", choices=["base", "voice_design", "cosyvoice3", "moss_local"])
    parser.add_argument("--voice", default="jason", help="Voice profile name (for base/cosyvoice3/moss_* mode)")
    args = parser.parse_args()

    segments = split_text(args.text)
    print(f"Split into {len(segments)} segment(s)")

    if args.mode in ("cosyvoice3", "moss_local"):
        voice_profile = VOICE_PROFILES.get(args.voice)
        if voice_profile is None:
            print(f"Warning: voice '{args.voice}' not found, using default")
            voice_profile = VOICE_PROFILES["jason"]

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
            print(f"  Generating segment {i+1}/{len(segments)}: {segment[:60]}...")
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
                if args.voice in VOICE_PROFILES:
                    kwargs.update(VOICE_PROFILES[args.voice])
                else:
                    print(f"Warning: voice '{args.voice}' not found, using default")
                    kwargs.update(VOICE_PROFILES["jason"])

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
