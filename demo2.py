import sys
from pathlib import Path
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

# ---- Config ----
MODE = sys.argv[1] if len(sys.argv) > 1 else "voice_design"

MODELS = {
    "voice_design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
}

VOICES_DIR = Path(__file__).parent / "voices"

VOICE_PROFILES = {
    "jason": {
        "ref_audio": str(VOICES_DIR / "jason.wav"),
        "ref_text": "大家好,我是Jason.欢迎回到我的频道.今天给大家讲一段Tesla的故事",
    },
}

TEXT = "哥哥，你回来啦，人家等了你好久好久了，要抱抱！"

INSTRUCT = "体现撒娇诱惑的女声，音调偏低，营造出黏人,勾引人,有欲望听觉效果。"

# ---- Run ----
if MODE not in MODELS:
    print(f"Usage: python demo2.py [voice_design|base]")
    sys.exit(1)

print(f"Mode: {MODE}")
model = load_model(MODELS[MODE])

kwargs = {
    "model": model,
    "text": TEXT,
    "file_prefix": f"output_{MODE}_demo2",
    "lang_code": "chinese",
}

if MODE == "voice_design":
    kwargs["instruct"] = INSTRUCT
elif MODE == "base":
    voice = sys.argv[2] if len(sys.argv) > 2 else "jason"
    if voice not in VOICE_PROFILES:
        print(f"Unknown voice: {voice}. Available: {', '.join(VOICE_PROFILES)}")
        sys.exit(1)
    kwargs.update(VOICE_PROFILES[voice])
    kwargs["file_prefix"] = f"output_{voice}_demo2"

generate_audio(**kwargs)
