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
        "大家好，我是 Jason。这笔 200 亿美元的 Cerebras 芯片交易，"
        "如果你跟着主流媒体看，你看到的是 OpenAI 的战略胜利——"
        "扩张版图、锁定供应链、和 Nvidia 分庭抗礼。"
        "但我越看越觉得，我和主流媒体的判断完全相反——"
        "这不是在向强，这是在用钱的动作挡焦虑。"
        "这期我用五个事实告诉你为什么。",
    ),
    (
        "02_revenue",
        "第一个事实是增速。"
        "OpenAI 营收从 2023 年 20 亿、到 2024 年 60 亿、再到 2025 年 200 亿，一年一个 3 倍。"
        "但 2026 年 The Information 披露的内部目标只有 294 亿——"
        "也就是 18% 到 47% 的增长，不管哪种口径都是显著放缓。"
        "更要命的是亏损——内部预测 2026 年亏 140 亿，盈亏平衡要等到 2030 年，"
        "而 Altman 已经承诺到 2033 年前烧 1.4 万亿美元建数据中心。"
        "增速从 3 倍掉到 18% 的同时承诺烧 1.4 万亿，这种数学只有一种解释——"
        "用增长叙事抢在基本面暴露前完成融资。",
    ),
    (
        "03_anthropic",
        "第二个事实——Anthropic 已经反超 OpenAI。"
        "但先说清楚一个口径问题，不然后面站不住脚。"
        "按各自口径，Anthropic 4 月 7 号官方宣布年化 300 亿，OpenAI 约 240 到 250 亿。"
        "但两家会计方式不同——OpenAI 云合作伙伴销售只计自己抽成，Anthropic 计全额，"
        "差异最多 80 亿。同口径比较，差距会缩小甚至反向。"
        "但真正不受口径争议的硬事实是这些："
        "企业新单 Anthropic 赢 70%，"
        "Claude Code 占企业 Coding 市场 54% 对 OpenAI Codex 的 21%，"
        "70% 的 Fortune 100 公司用 Claude。"
        "同口径，可审计。再加上 Anthropic 训练成本只有 OpenAI 的四分之一。"
        "当主流叙事跟同口径数据错位到这种程度，那主流叙事就是过时的。",
    ),
    (
        "04_internal",
        "第三个事实是内部信号——"
        "当一家公司的 CFO 被 CEO 排除在投资者会议外，基本面已经不是秘密。"
        "OpenAI CFO Sarah Friar，2025 年 8 月起不再向 Altman 直接汇报。"
        "Altman 把她排除在投资者会议外——这是 The Information 和 Fortune 独立确认的。"
        "她明确反对 2026 Q4 IPO 时间表，质疑公司能否支撑 6000 亿美元的支出承诺。"
        "CFO 公开反对 CEO 的核心战略在公司治理史上都极罕见——"
        "她做了，说明内部裂痕比外面能看到的深得多。"
        "再看人才层面。OpenAI 联合创始人 John Schulman，后训练方向核心研究员——"
        "离开后跑去了 Anthropic。不是创业，不是退休，是去了最大竞争对手。"
        "这比任何新闻稿都真实——"
        "当最懂模型的人都选择跳槽到对手那里，内部故事就讲不圆了。"
        "加上 2025 年 12 名高管离职，11 位创始人剩 2 位——Brockman 也卸任了 chairman。"
        "这些事叠在一起，回头再看那笔 200 亿交易，味道就完全不一样了。",
    ),
    (
        "05_anxiety",
        "同一时间窗，OpenAI 还在做什么？"
        "新一轮 1220 亿融资——Amazon 500 亿、Nvidia 300 亿、SoftBank 300 亿。"
        "估值两周内从 8300 亿跳到 8520 亿——220 亿美元增量，两周。"
        "Altman 推 Q4 IPO 的理由？抢在 Anthropic 前面。"
        "把这些动作串起来：1220 亿融资、估值两周跳 220 亿、Cerebras 200 亿深绑、"
        "IPO 抢跑、CFO 反对——不是扩张节奏，是焦虑节奏。"
        "稳健的公司不会在两周内把估值拉起来 220 亿。"
        "真的在赢的公司不会抢着 IPO、不会把 CFO 排除在外、"
        "不会让联合创始人跑去对手那里。"
        "单独看 Cerebras 是战略胜利，放回这组动作里，就是焦虑症状的一次放大。",
    ),
    (
        "06_defuse",
        "现在正面回应三个反驳。"
        "第一个反驳：ChatGPT 周活 9 亿还在翻倍增长，基本面明明很强。"
        "我的回应——用户增长不等于营收增长。"
        "C 端 ARPU 太低，利润大头在企业端，而企业端 Anthropic 已经用同口径数据反超。"
        "用户数是必要条件，不是充分条件——Snapchat 教过这一课。"
        "第二个反驳：1220 亿融资有 Amazon、Nvidia、SoftBank 下注，说明巨头认账。"
        "我的回应——这恰恰是循环融资的加深。"
        "举最露骨的例子——Nvidia 投 OpenAI 的 1000 亿，"
        "Nvidia 自己的账面价值直接挂钩 OpenAI 估值，它不是独立下注，"
        "是把自己卖给 OpenAI 的 GPU 收入记回了自己账面。"
        "SoftBank 通过 Stargate 锁定 5000 亿算力承诺，它退不出去。"
        "Amazon 是争 AWS 云份额。投你的客户不是投你，是投自己账面。"
        "第三个反驳：不断融资扩张是创业公司常态。"
        "我的回应——时机和节奏不正常。"
        "CFO 反对、高管流失、估值两周跳 220 亿，都不是常态参数。"
        "平时不加班突然全员加班，不能用加班很常见来解释。"
        "稳态扩张从不需要两周跳 220 亿估值。"
        "但我的判断不是 OpenAI 要完了——这里要分清楚——"
        "是 OpenAI 在用钱的动作挡焦虑。"
        "这是风险信号，不是崩盘预言。"
        "区分这两者，是看财经新闻最重要的能力之一。",
    ),
    (
        "07_outro",
        "最后留三个可以跟踪的验证信号："
        "OpenAI Q2 2026 财报——增速从 18% 反弹到 40%，我的判断瓦解；"
        "Friar 的去留——离职或和 Altman 公开和解，信息含量都很大；"
        "Q4 IPO 能不能按 8520 亿估值上市——成了说明市场仍买账，黄了是 thesis 最强验证。"
        "给你留个赌局——Q4 IPO 能不能按 8520 亿估值成功？"
        "评论区留 bull 或 bear，你押哪边？"
        "完整研究报告和 fact cards 都在置顶评论，点赞订阅，"
        "下期聊 Cerebras 350 亿 IPO 估值合不合理。下期见。",
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
                    "volume=2.7",
                    str(out_path),
                ],
                capture_output=True,
                check=True,
            )
            print(f"  -> {out_path}")

    print("\nAll done. Output dir:", OUT_DIR)


if __name__ == "__main__":
    main()
