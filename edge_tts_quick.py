#!/usr/bin/env python3
"""
Fast TTS using Microsoft Edge TTS (free, no API key needed).
Much faster than MLX TTS for Chinese text.
"""
import asyncio
import edge_tts
import os
from pathlib import Path

TEXT_FILE = Path(__file__).parent / "input" / "sonnet46.txt"
OUTPUT_FILE = Path(__file__).parent / "output" / "narration.wav"
VOICE = "zh-CN-XiaoxiaoNeural"  # Popular Chinese female voice

async def tts_edge(text: str, output_path: str):
    """Generate TTS using Edge TTS."""
    communicate = edge_tts.Communicate(text, VOICE)
    await communicate.save(output_path)

async def main():
    if not TEXT_FILE.exists():
        print(f"Error: {TEXT_FILE} not found")
        return

    text = TEXT_FILE.read_text(encoding="utf-8").strip()
    if not text:
        print("Empty file, skipping")
        return

    print(f"Generating TTS for {len(text)} characters using {VOICE}...")

    # Generate audio
    await tts_edge(text, str(OUTPUT_FILE))

    print(f"Done! Saved to {OUTPUT_FILE}")
    print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    asyncio.run(main())
