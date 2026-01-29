from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
generate_audio(
    model=model,
    text='''朋友们，今天这份财报，即便你不是特斯拉的股东，我也建议你听一听。因为在2026年1月28日这一天，我们熟悉的那个“车企特斯拉”，正式从ICU被推向了这一页历史的边缘。

为什么这么说？

这次财报会上，Elon Musk 轻描淡写地丢下了一颗重磅炸弹——Model S 和 Model X，这两款定义了什么是“豪华电动车”的旗舰车型，将在下个季度正式“光荣退役”。并不是更新换代，而是直接停产。

而原本在 Fremont 工厂生产这两款车的生产线，将被全部拆除，腾出来的地方要造什么？不是更便宜的车，不是 Model 2，而是——Optimus 人形机器人。

Elon 说："It is time to bring the S and X programs to an end."（是时候结束S和X项目了）。

这不仅仅是一个产品线的调整，这是一封“分手信”。特斯拉正在跟它过去赖以生存的“卖车赚钱”的商业模式彻底分手。

很多人看完财报的第一反应是：利润暴跌、营收负增长、比亚迪骑脸输出。完了，特斯拉甚至有了那种“诺基亚时刻”的衰败感。
但你要是只看到这层，那你可能就真的错过了下一个十年最大的赌局。''',
    instruct="A cheerful young female voice with clear pronunciation and moderate speed",
    file_prefix="test_audio",
)
