"""
Batch process txt files in input/ directory.
- Scans for unprocessed .txt files (no "finished_" prefix)
- Splits long text into paragraphs and generates TTS for each
- Merges all segments into one wav file
- Renames file to "finished_<original>" after processing
"""

import re
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.generate import generate_audio, load_audio
from mlx_audio.tts.utils import load_model

# ---- Config ----
INPUT_DIR = Path(__file__).parent / "input"
OUTPUT_DIR = Path(__file__).parent / "output"
VOICES_DIR = Path(__file__).parent / "voices"

MODELS = {
    "voice_design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
}

VOICE_PROFILES = {
    "jason": {
        "ref_audio": str(VOICES_DIR / "jason.wav"),
        "ref_text": "大家好,我是Jason.欢迎回到我的频道.今天给大家讲一段Tesla的故事",
    },
}

DEFAULT_MODE = "base"
DEFAULT_VOICE = "jason"
DEFAULT_INSTRUCT = "A cheerful young female voice with clear pronunciation and moderate speed"


MAX_SEGMENT_CHARS = 200


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences at natural boundaries."""
    # Split at Chinese/English sentence-ending punctuation, keeping the punctuation
    parts = re.split(r"(?<=[。！？；\?\!])", text)
    return [p.strip() for p in parts if p.strip()]


def split_text(text: str) -> list[str]:
    """Split text into segments that respect sentence boundaries.

    1. Split by blank lines into paragraphs
    2. Split long paragraphs into sentences
    3. Group short sentences together up to MAX_SEGMENT_CHARS
    """
    # Split by blank lines
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip().replace("\n", "") for p in paragraphs]
    paragraphs = [p for p in paragraphs if p and p.strip("-— ") != ""]

    segments: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) <= MAX_SEGMENT_CHARS:
            # Short paragraph: try to group with buffer
            if buffer and len(buffer) + len(para) > MAX_SEGMENT_CHARS:
                segments.append(buffer)
                buffer = para
            else:
                buffer = f"{buffer}{para}" if buffer else para
        else:
            # Long paragraph: flush buffer first, then split into sentences
            if buffer:
                segments.append(buffer)
                buffer = ""
            for sentence in split_into_sentences(para):
                if buffer and len(buffer) + len(sentence) > MAX_SEGMENT_CHARS:
                    segments.append(buffer)
                    buffer = sentence
                else:
                    buffer = f"{buffer}{sentence}" if buffer else sentence

    if buffer:
        segments.append(buffer)

    return segments


def get_pending_files() -> list[Path]:
    """Get txt files that haven't been processed yet."""
    return sorted(
        f for f in INPUT_DIR.glob("*.txt") if not f.name.startswith("finished_")
    )


def process_file(filepath: Path, model, mode: str, voice: str = DEFAULT_VOICE):
    """Process a single txt file: split text, TTS each segment, merge, rename.

    Each segment is saved as an individual wav file. If interrupted,
    re-running will skip segments that already have a wav file on disk.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {filepath.name} (mode={mode})")
    print(f"{'='*60}")

    text = filepath.read_text(encoding="utf-8").strip()
    if not text:
        print("  Skipped (empty file)")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    segments = split_text(text)
    stem = filepath.stem
    print(f"  Split into {len(segments)} segment(s)")

    # Generate each segment, skipping existing ones
    for i, segment in enumerate(segments):
        seg_file = OUTPUT_DIR / f"{stem}_seg_{i:03d}.wav"

        if seg_file.exists():
            print(f"\n  --- Segment {i+1}/{len(segments)} --- SKIP (already exists)")
            continue

        print(f"\n  --- Segment {i+1}/{len(segments)} ---")
        print(f"  {segment[:80]}...")

        seg_prefix = str(OUTPUT_DIR / f"{stem}_seg_{i:03d}")

        kwargs = {
            "model": model,
            "text": segment,
            "file_prefix": seg_prefix,
            "join_audio": True,
            "max_tokens": 4096,
            "lang_code": "chinese",
        }

        if mode == "voice_design":
            kwargs["instruct"] = DEFAULT_INSTRUCT
        elif mode == "base":
            kwargs.update(VOICE_PROFILES[voice])

        generate_audio(**kwargs)

    # Merge all segment files into one
    seg_files = sorted(OUTPUT_DIR.glob(f"{stem}_seg_*.wav"))
    if not seg_files:
        print("  No audio generated")
        return

    print(f"\n  Merging {len(seg_files)} segment(s)...")
    all_audio = [load_audio(str(f), sample_rate=model.sample_rate) for f in seg_files]
    merged = mx.concatenate(all_audio, axis=0)
    output_file = str(OUTPUT_DIR / f"{stem}.wav")
    audio_write(output_file, np.array(merged), model.sample_rate, format="wav")

    # Boost volume by 80% (Qwen3-TTS output is too quiet)
    boosted_file = str(OUTPUT_DIR / f"{stem}_boosted.wav")
    import subprocess as _sp
    _sp.run(["ffmpeg", "-y", "-i", output_file, "-filter:a", "volume=1.8", boosted_file],
            capture_output=True)
    if Path(boosted_file).exists():
        Path(output_file).unlink()
        Path(boosted_file).rename(output_file)
        print(f"  -> {output_file} (volume boosted 1.8x)")
    else:
        print(f"  -> {output_file} (volume boost failed, using original)")

    # Clean up segment files
    for f in seg_files:
        f.unlink()
    print(f"  Cleaned up {len(seg_files)} segment file(s)")

    # Rename to mark as finished
    finished_path = filepath.parent / f"finished_{filepath.name}"
    filepath.rename(finished_path)
    print(f"  Done -> {finished_path.name}")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODE

    if mode not in MODELS:
        print(f"Usage: python process.py [voice_design|base] [voice_name]")
        print(f"  Modes: {', '.join(MODELS)}")
        print(f"  Voices (base mode only): {', '.join(VOICE_PROFILES)}")
        sys.exit(1)

    voice = DEFAULT_VOICE
    if mode == "base" and len(sys.argv) > 2:
        voice = sys.argv[2]
        if voice not in VOICE_PROFILES:
            print(f"Unknown voice: {voice}. Available: {', '.join(VOICE_PROFILES)}")
            sys.exit(1)

    pending = get_pending_files()
    if not pending:
        print("No pending txt files in input/")
        return

    print(f"Found {len(pending)} file(s) to process:")
    for f in pending:
        print(f"  - {f.name}")

    model_name = MODELS[mode]
    print(f"\nLoading model: {model_name} (mode={mode})")
    model = load_model(model_name)

    for filepath in pending:
        process_file(filepath, model, mode, voice)

    print(f"\nAll done! Processed {len(pending)} file(s).")


if __name__ == "__main__":
    main()
