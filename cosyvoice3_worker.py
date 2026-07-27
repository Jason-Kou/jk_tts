#!/usr/bin/env python3
"""CosyVoice3 worker process.

Run this with the CosyVoice Python environment, not the jk_tts uv env.
It loads Fun-CosyVoice3 once, then synthesizes all requested segments.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torchaudio


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TTS with Fun-CosyVoice3")
    parser.add_argument("--request", required=True, help="Path to request JSON")
    args = parser.parse_args()

    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding="utf-8"))

    cosyvoice_repo = Path(request["cosyvoice_repo"]).expanduser().resolve()
    sys.path.insert(0, str(cosyvoice_repo))
    sys.path.append(str(cosyvoice_repo / "third_party" / "Matcha-TTS"))

    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

    model_dir = Path(request["model_dir"]).expanduser().resolve()
    prompt_audio = Path(request["prompt_audio"]).expanduser().resolve()
    prompt_text = request["prompt_text"]
    jobs = request["jobs"]

    print(
        f"[CosyVoice3] torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()} mps={torch.backends.mps.is_available()}"
    )
    print(f"[CosyVoice3] model={model_dir}")
    print(f"[CosyVoice3] prompt_audio={prompt_audio}")

    load_started = time.perf_counter()
    model = AutoModel(model_dir=str(model_dir))
    print(f"[CosyVoice3] model_loaded_s={time.perf_counter() - load_started:.2f}")

    results = []
    for index, job in enumerate(jobs, start=1):
        text = job["text"]
        output = Path(job["output"]).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)

        print(f"[CosyVoice3] segment {index}/{len(jobs)} chars={len(text)}")
        gen_started = time.perf_counter()
        chunks = list(
            model.inference_zero_shot(
                text,
                prompt_text,
                str(prompt_audio),
                stream=False,
            )
        )
        gen_s = time.perf_counter() - gen_started
        if not chunks:
            raise RuntimeError(f"CosyVoice3 returned no audio for segment {index}")

        speech = torch.cat([chunk["tts_speech"] for chunk in chunks], dim=1)
        torchaudio.save(str(output), speech, model.sample_rate)
        duration_s = speech.shape[1] / float(model.sample_rate)
        print(
            f"[CosyVoice3] wrote={output} duration_s={duration_s:.2f} "
            f"generation_s={gen_s:.2f} rtf={gen_s / duration_s if duration_s else 0:.2f}"
        )
        results.append(
            {
                "output": str(output),
                "sample_rate": model.sample_rate,
                "duration_s": duration_s,
                "generation_s": gen_s,
            }
        )

    print(json.dumps({"ok": True, "results": results}, ensure_ascii=False))


if __name__ == "__main__":
    main()
