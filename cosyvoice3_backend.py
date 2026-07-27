"""Helpers for using Fun-CosyVoice3 as a jk_tts backend."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).parent
DEFAULT_COSYVOICE_REPO = Path(os.getenv("COSYVOICE_REPO", "/Users/jk-agent-mac/3_coding/CosyVoice"))
DEFAULT_COSYVOICE_PYTHON = Path(
    os.getenv("COSYVOICE_PYTHON", str(DEFAULT_COSYVOICE_REPO / ".venv-cosy" / "bin" / "python"))
)
DEFAULT_COSYVOICE3_MODEL_DIR = Path(
    os.getenv("COSYVOICE3_MODEL_DIR", str(DEFAULT_COSYVOICE_REPO / "pretrained_models" / "Fun-CosyVoice3-0.5B"))
)
DEFAULT_PROMPT_PREFIX = "You are a helpful assistant.<|endofprompt|>"


def _check_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def cosyvoice3_available() -> bool:
    return (
        DEFAULT_COSYVOICE_REPO.exists()
        and DEFAULT_COSYVOICE_PYTHON.exists()
        and DEFAULT_COSYVOICE3_MODEL_DIR.exists()
    )


def synthesize_segments(
    segments: list[str],
    output_files: list[Path],
    voice_profile: dict[str, str],
) -> None:
    """Generate segment WAV files with Fun-CosyVoice3.

    The heavy CosyVoice dependencies live in a separate environment, so this
    function shells out to `cosyvoice3_worker.py` using that environment.
    """
    if len(segments) != len(output_files):
        raise ValueError("segments and output_files must have the same length")

    _check_path(DEFAULT_COSYVOICE_REPO, "CosyVoice repository")
    _check_path(DEFAULT_COSYVOICE_PYTHON, "CosyVoice Python")
    _check_path(DEFAULT_COSYVOICE3_MODEL_DIR, "CosyVoice3 model directory")

    prompt_audio = Path(voice_profile["ref_audio"])
    _check_path(prompt_audio, "CosyVoice3 prompt audio")
    prompt_text = DEFAULT_PROMPT_PREFIX + voice_profile["ref_text"]

    worker = ROOT / "cosyvoice3_worker.py"
    _check_path(worker, "CosyVoice3 worker")

    jobs = [
        {"text": segment, "output": str(output_file)}
        for segment, output_file in zip(segments, output_files)
    ]
    request = {
        "cosyvoice_repo": str(DEFAULT_COSYVOICE_REPO),
        "model_dir": str(DEFAULT_COSYVOICE3_MODEL_DIR),
        "prompt_audio": str(prompt_audio),
        "prompt_text": prompt_text,
        "jobs": jobs,
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as f:
        json.dump(request, f, ensure_ascii=False)
        request_path = Path(f.name)

    try:
        subprocess.run(
            [str(DEFAULT_COSYVOICE_PYTHON), str(worker), "--request", str(request_path)],
            cwd=str(ROOT),
            check=True,
        )
    finally:
        request_path.unlink(missing_ok=True)


def merge_wavs_with_ffmpeg(seg_files: list[Path], output: Path, volume: float = 1.0) -> None:
    """Merge same-format WAV files and optionally apply a volume multiplier."""
    if not seg_files:
        raise ValueError("No segment files to merge")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        concat_file = tmpdir_path / "segments.txt"
        merged_file = tmpdir_path / "merged.wav"
        concat_file.write_text(
            "".join(f"file '{path.resolve()}'\n" for path in seg_files),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-c",
                "copy",
                str(merged_file),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if volume == 1.0:
            output.write_bytes(merged_file.read_bytes())
            return

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(merged_file),
                "-filter:a",
                f"volume={volume}",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
