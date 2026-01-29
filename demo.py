import sys
from pathlib import Path
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

# ---- Config ----
MODE = sys.argv[1] if len(sys.argv) > 1 else "base"

MODELS = {
    "voice_design": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16",
    "base": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
}

VOICES_DIR = Path(__file__).parent / "voices"

# Voice profiles: audio file + transcript
VOICE_PROFILES = {
    "jason": {
        "ref_audio": str(VOICES_DIR / "jason.wav"),
        "ref_text": "大家好,我是Jason.欢迎回到我的频道.今天给大家讲一段Tesla的故事",
    },
    # Add more voices here:
    # "alice": {
    #     "ref_audio": str(VOICES_DIR / "alice.wav"),
    #     "ref_text": "transcript of alice's reference audio",
    # },
}

TEXT = """朋友们，今天这份财报，即便你不是特斯拉的股东，我也建议你听一听。因为在2026年1月28日这一天，我们熟悉的那个"车企特斯拉"，正式从ICU被推向了这一页历史的边缘。

为什么这么说？

这次财报会上，Elon Musk 轻描淡写地丢下了一颗重磅炸弹——Model S 和 Model X，这两款定义了什么是"豪华电动车"的旗舰车型，将在下个季度正式"光荣退役"。并不是更新换代，而是直接停产。

而原本在 Fremont 工厂生产这两款车的生产线，将被全部拆除，腾出来的地方要造什么？不是更便宜的车，不是 Model 2，而是——Optimus 人形机器人。

Elon 说："It is time to bring the S and X programs to an end."（是时候结束S和X项目了）。

这不仅仅是一个产品线的调整，这是一封"分手信"。特斯拉正在跟它过去赖以生存的"卖车赚钱"的商业模式彻底分手。

很多人看完财报的第一反应是：利润暴跌、营收负增长、比亚迪骑脸输出。完了，特斯拉甚至有了那种"诺基亚时刻"的衰败感。
但你要是只看到这层，那你可能就真的错过了下一个十年最大的赌局。"""

# ---- Run ----
if MODE not in MODELS:
    print(f"Usage: python demo.py [voice_design|base]")
    sys.exit(1)

print(f"Mode: {MODE}")
model = load_model(MODELS[MODE])

kwargs = {
    "model": model,
    "text": TEXT,
    "file_prefix": f"output_{MODE}",
}

if MODE == "voice_design":
    kwargs["instruct"] = "A cheerful young female voice with clear pronunciation and moderate speed"
elif MODE == "base":
    voice = sys.argv[2] if len(sys.argv) > 2 else "jason"
    if voice not in VOICE_PROFILES:
        print(f"Unknown voice: {voice}. Available: {', '.join(VOICE_PROFILES)}")
        sys.exit(1)
    kwargs.update(VOICE_PROFILES[voice])
    kwargs["file_prefix"] = f"output_{voice}"

generate_audio(**kwargs)
