#!/usr/bin/env python3
"""
Batch-generate 7 narration WAVs for the Cerebras video.
One model load; each scene gets its own .wav with audio timing we can measure.
"""
import sys
import subprocess
import tempfile
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.generate import generate_audio, load_audio
from mlx_audio.tts.utils import load_model

VOICES_DIR = Path(__file__).parent / "voices"
OUT_DIR = Path(
    "/Users/jk-agent-mac/3_coding/jk_agent/output/cerebras_20260420/narration"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
REF_AUDIO = str(VOICES_DIR / "jason.wav")
REF_TEXT = "大家好,我是Jason.欢迎回到我的频道.今天给大家讲一段Tesla的故事"

MAX_SEG_CHARS = 200


SEGMENTS = [
    (
        "01_intro",
        "大家好，我是 Jason。今天聊一个很有意思的数字——200亿美元。"
        "上周 OpenAI 官宣，要花 200 亿买 Cerebras 的 AI 芯片。听起来就是个大订单？"
        "但你要是仔细看合同细节，会发现三件事："
        "第一，OpenAI 顺便拿了 Cerebras 10% 的股份；"
        "第二，Cerebras 同一天递交了 IPO 招股书，目标估值 350 亿；"
        "第三，OpenAI 还给 Cerebras 借了 10 亿美元去建数据中心。"
        "注意了——OpenAI 同时是它最大的客户、最大的股东，和债主。"
        "这已经不是买芯片了，这是把上游供应商整个『收编』。",
    ),
    (
        "02_triple",
        "先把这笔交易的结构说清楚。"
        "2026 年 1 月，OpenAI 先签了个 100 亿的合同，买 Cerebras 750 兆瓦的算力，用到 2028 年。"
        "到了 4 月 17 号，也就是上周，这份合同直接翻倍，变成 200 亿美元，"
        "追加 1.25 吉瓦的算力选项，一直到 2030 年。"
        "关键是除了采购，OpenAI 还拿了两个东西。"
        "一个是 Cerebras 的股票认购权证，采购达标就触发，最高能占到 Cerebras 全部股份的 10%；"
        "另一个是 10 亿美元的数据中心建设贷款，年利 6%。"
        "Cerebras 拿这笔钱去建厂，建好的厂专门跑 OpenAI 的负载，"
        "OpenAI 的股权价值又跟着 Cerebras 的估值一起涨——然后 Cerebras 正好在这个时间点 IPO。"
        "这个闭环太漂亮了。",
    ),
    (
        "03_hardware",
        "这一单看着吓人，但你放到 OpenAI 整个硬件版图里，会发现这只是最新一块拼图。"
        "我们来盘点一下：Nvidia 那边，2025 年 9 月签了 10 吉瓦的战略合作，Nvidia 反向投资 OpenAI 最多 1000 亿美元；"
        "AMD 那边，签了 6 吉瓦的合同，AMD 给 OpenAI 发了 1.6 亿股的股票权证；"
        "Broadcom 那边，OpenAI 跟它一起做自研 ASIC 芯片，规模也是 10 吉瓦，2026 年下半年量产；"
        "现在加上 Cerebras 的 1.25 吉瓦。"
        "算下来，OpenAI 锁定的算力已经超过 27 吉瓦，投入的资本承诺，保守估计超过 5000 亿美元。"
        "而且你看这个套路——每家都是『采购合同加股权权证』的组合拳。"
        "OpenAI 已经不是一家模型公司了，它是 AI 硬件生态里最大的战略投资方。",
    ),
    (
        "04_perf",
        "有人可能会问：都有 Nvidia 了，为什么还要 Cerebras？"
        "因为 Nvidia GPU 在推理这一侧，性价比已经不够好了。"
        "Cerebras 的核心武器叫 WSE-3，把一整片 300 毫米晶圆做成单个芯片，4 万亿个晶体管，90 万核心。"
        "内存带宽是 Nvidia H100 的 7000 倍——注意，是 7000 倍。"
        "这个数字直接决定了大模型推理的速度上限。"
        "具体数据：跑 Llama 3.1 的 405B 模型，Cerebras 能达到 969 tokens 每秒，是主流云 GPU 的 75 倍；"
        "跑 GPT-OSS 120B，单用户速度 3098 tokens 每秒。"
        "OpenAI 最新的编码模型 GPT-5.3-Codex-Spark，2 月份官宣了 1000 tokens 每秒——就跑在 Cerebras 的 WSE-3 上。"
        "这是 OpenAI 第一次把产品模型部署在非 Nvidia 的硬件上。信号已经发出来了。",
    ),
    (
        "05_controversy",
        "但这事没那么光鲜。最大的争议叫『循环融资』。"
        "你想想这个闭环——OpenAI 投钱给 Cerebras，Cerebras 用这笔钱建厂跑 OpenAI 的活，"
        "OpenAI 股权跟着 Cerebras 估值涨，最后 Cerebras 带着 OpenAI 的订单去 IPO——"
        "钱其实没怎么真正流出过这个圈子。"
        "科技评论人 Ed Zitron 在 X 上直接开喷，说这叫『股票操纵，帮 Cerebras IPO 定高价』。"
        "Seaport 的分析师 Jay Goldberg 说这整件事『非常模糊——分不清 Nvidia 是投资 OpenAI 还是在补贴自己的需求』。"
        "Bernstein 的 Stacy Rasgon 更直接：这种结构『必然会放大市场对循环融资的担忧』。"
        "再看 Cerebras 自己的财务质量——招股书显示 2025 年收入 5.1 亿美元，"
        "但报表里的净利润包含了 3.91 亿美元的『其他收入』，剔掉之后实际经营净亏损 1.53 亿。"
        "更要命的是客户集中风险：两个大客户占了 86% 的收入。"
        "再加上 WSE-3 独家 TSMC 5 纳米制造——TSMC 一打喷嚏，OpenAI 的合同立马违约。",
    ),
    (
        "06_strategy",
        "我个人怎么看这件事？我觉得 Altman 比外界意识到的要老练得多。"
        "2024 年 2 月他喊过要融 5 到 7 万亿美元建全球芯片代工网络，当时所有人都觉得他疯了。"
        "两年过去，他没自己造芯片，但他把造芯片的人都变成了自己的股东。"
        "Nvidia、AMD、Broadcom、Cerebras——每一家都跟 OpenAI 股权绑定。"
        "这是从『我要造芯片挑战 Nvidia』升级到『我用金融结构把整个供应链锁起来』。"
        "这玩法确实高明，但风险也更大——所有人都押在同一张牌上，"
        "而这张牌叫『AI 推理需求会继续 10 倍量级增长』。"
        "如果哪天需求曲线拐弯了，整个循环立刻反向。这是 2000 年互联网泡沫『供应商融资』模式的放大版。",
    ),
    (
        "07_outro",
        "最后留几个悬念。"
        "第一，这 200 亿是强制采购还是弹性承诺？合同细节到现在没公开。"
        "第二，OpenAI 自己的 Broadcom 自研芯片 2026 下半年量产以后，会不会砍 Cerebras 的订单？"
        "第三，对中国 AI 产业的启示——头部模型公司绑定上游芯片的战略已经打响。"
        "这期就到这。完整的研究报告我都放在置顶评论里了。"
        "如果这种深度分析对你有用，点个赞和订阅。"
        "下期我们聊 Broadcom 的自研 ASIC——OpenAI 这盘棋最关键的一颗子。我们下期见。",
    ),
]


def split_text(text: str) -> list[str]:
    import re

    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [p.strip().replace("\n", "") for p in paragraphs if p.strip()]

    segments = []
    buffer = ""
    for para in paragraphs:
        if len(para) <= MAX_SEG_CHARS:
            if buffer and len(buffer) + len(para) > MAX_SEG_CHARS:
                segments.append(buffer)
                buffer = para
            else:
                buffer = f"{buffer}{para}" if buffer else para
        else:
            if buffer:
                segments.append(buffer)
                buffer = ""
            parts = re.split(r"(?<=[。！？；\?\!])", para)
            for sentence in [p.strip() for p in parts if p.strip()]:
                if buffer and len(buffer) + len(sentence) > MAX_SEG_CHARS:
                    segments.append(buffer)
                    buffer = sentence
                else:
                    buffer = f"{buffer}{sentence}" if buffer else sentence
    if buffer:
        segments.append(buffer)
    return segments


def main() -> None:
    print(f"Loading model: {MODEL_ID}")
    model = load_model(MODEL_ID)
    print("Model loaded.")

    for slug, text in SEGMENTS:
        out_path = OUT_DIR / f"{slug}.wav"
        if out_path.exists():
            print(f"[SKIP] {slug} (already exists)")
            continue

        pieces = split_text(text)
        print(f"[{slug}] {len(text)} chars → {len(pieces)} piece(s)")

        with tempfile.TemporaryDirectory() as tmpdir:
            piece_files = []
            for i, piece in enumerate(pieces):
                print(f"  piece {i + 1}/{len(pieces)}: {piece[:40]}...")
                prefix = str(Path(tmpdir) / f"{slug}_{i:03d}")
                generate_audio(
                    model=model,
                    text=piece,
                    file_prefix=prefix,
                    join_audio=True,
                    max_tokens=4096,
                    lang_code="chinese",
                    ref_audio=REF_AUDIO,
                    ref_text=REF_TEXT,
                )
                f = Path(f"{prefix}.wav")
                if f.exists():
                    piece_files.append(f)

            if not piece_files:
                print(f"  ERROR: no audio for {slug}")
                continue

            arrs = [
                load_audio(str(f), sample_rate=model.sample_rate)
                for f in piece_files
            ]
            merged = mx.concatenate(arrs, axis=0)

            tmp_wav = str(Path(tmpdir) / "merged.wav")
            audio_write(
                tmp_wav, np.array(merged), model.sample_rate, format="wav"
            )

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_wav,
                    "-filter:a",
                    "volume=1.8",
                    str(out_path),
                ],
                capture_output=True,
                check=True,
            )
            print(f"  -> {out_path}")

    print("\nAll done. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
