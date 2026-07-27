"""Helpers for running MOSS-TTS through a separate MLX-Audio environment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DEFAULT_MOSS_PYTHON = Path(os.getenv("MOSS_TTS_PYTHON", str(ROOT / ".venv-moss" / "bin" / "python")))
DEFAULT_MOSS_NANO_MODEL = os.getenv("MOSS_TTS_NANO_MODEL", "mlx-community/MOSS-TTS-Nano-100M")
DEFAULT_MOSS_LOCAL_MODEL = os.getenv("MOSS_TTS_LOCAL_MODEL", "OpenMOSS-Team/MOSS-TTS-Local-Transformer")
DEFAULT_MOSS_LANG_CODE = os.getenv("MOSS_TTS_LANG_CODE", "zh")


def moss_model_for_mode(mode: str) -> str:
    if mode == "moss_nano":
        return DEFAULT_MOSS_NANO_MODEL
    if mode == "moss_local":
        return DEFAULT_MOSS_LOCAL_MODEL
    raise ValueError(f"unsupported MOSS mode: {mode}")


def _check_path(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def synthesize_segments(
    segments: Iterable[str],
    output_files: Iterable[Path],
    voice_profile: dict[str, str],
    model_name: str,
    *,
    max_tokens: int = 4096,
) -> None:
    _check_path(DEFAULT_MOSS_PYTHON, "MOSS-TTS Python")

    prompt_audio = Path(voice_profile["ref_audio"])
    _check_path(prompt_audio, "MOSS-TTS reference audio")
    prompt_text = voice_profile["ref_text"]

    for index, (segment, output_file) in enumerate(zip(segments, output_files), start=1):
        output_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"[MOSS-TTS] segment {index} chars={len(segment)} model={model_name}")

        cmd = [
            str(DEFAULT_MOSS_PYTHON),
            "-m",
            "mlx_audio.tts.generate",
            "--model",
            model_name,
            "--text",
            segment,
            "--lang_code",
            DEFAULT_MOSS_LANG_CODE,
            "--ref_audio",
            str(prompt_audio),
            "--ref_text",
            prompt_text,
            "--output_path",
            str(output_file.parent),
            "--file_prefix",
            output_file.stem,
            "--join_audio",
            "--max_tokens",
            str(max_tokens),
            "--verbose",
        ]
        result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        if result.stdout:
            print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        if result.returncode != 0:
            raise RuntimeError(f"MOSS-TTS generation failed with exit code {result.returncode}")
        if not output_file.exists() or output_file.stat().st_size == 0:
            raise RuntimeError(f"MOSS-TTS did not create output file: {output_file}")
