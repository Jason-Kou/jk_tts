# jk-tts

基于 [mlx-audio](https://github.com/lucasnewman/mlx-audio) 和 [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign) 的 Apple Silicon 本地 TTS 项目。

## 环境要求

- macOS (Apple Silicon)
- Python 3.10 ~ 3.12（**不支持 3.13+**，因为 `mlx-audio` 0.3.0 依赖的 `transformers==5.0.0rc3` 尚未兼容）
- [uv](https://docs.astral.sh/uv/) 包管理器

## 安装

```bash
# 克隆项目
git clone <repo-url> && cd jk_tts

# 安装依赖（uv 会自动创建虚拟环境并处理 pre-release 依赖）
uv sync
```

首次运行时，模型权重（约 4.5 GB）会自动从 HuggingFace 下载并缓存到 `~/.cache/huggingface/`。

## 模型说明

项目支持两种 Qwen3-TTS 模型，通过命令行参数切换：

| 模式 | 模型 | 功能 | 适用场景 |
|------|------|------|----------|
| `base`（默认） | Qwen3-TTS-Base | 声音克隆：3 秒参考音频即可复刻音色 | 用自己的声音生成语音 |
| `voice_design` | Qwen3-TTS-VoiceDesign | 文字描述生成声音（无需参考音频） | 自由设计声音风格 |
| `cosyvoice3` | Fun-CosyVoice3-0.5B | 本地 PyTorch 后端，中文中长旁白稳定性测试 | 与 Qwen3-TTS 做视频旁白效果对比 |
| `moss_nano` | MOSS-TTS-Nano-100M | MLX 后端，48 kHz 立体声，Mac 上速度快、内存低 | 推荐的 MOSS-TTS Mac 测试模式 |
| `moss_local` | MOSS-TTS-Local-Transformer | MLX 后端，1.7B，更正式但很慢、占内存高 | 质量对比或离线慢速测试 |

`cosyvoice3` 模式默认复用本机 `/Users/jk-agent-mac/3_coding/CosyVoice` 中已下载的 Fun-CosyVoice3 模型和独立 Python 环境；可用 `COSYVOICE_REPO`、`COSYVOICE_PYTHON`、`COSYVOICE3_MODEL_DIR` 覆盖路径。

`moss_nano` / `moss_local` 模式默认使用本项目的独立环境 `.venv-moss`，避免影响现有 Qwen3-TTS 依赖；可用 `MOSS_TTS_PYTHON`、`MOSS_TTS_NANO_MODEL`、`MOSS_TTS_LOCAL_MODEL` 覆盖路径或模型。MOSS 模式需要参考音频，推荐先用 `official_female`。

## 使用

### 声音克隆（Base 模式，默认）

```bash
# 使用默认声音 jason
uv run python demo.py

# 指定声音
uv run python demo.py base jason
```

输出文件：`output_jason_000.wav`

### VoiceDesign 模式

```bash
uv run python demo.py voice_design
```

输出文件：`output_voice_design_000.wav`

### CLI

```bash
# VoiceDesign
uv run python -m mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 \
  --text "Hello, this is a test." \
  --instruct "A calm male voice with low pitch"

# Base (voice cloning)
uv run python -m mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16 \
  --text "Hello, this is a test." \
  --ref-audio voices/jason.wav \
  --ref-text "大家好,我是Jason.欢迎回到我的频道.今天给大家讲一段Tesla的故事"

# Fun-CosyVoice3 comparison backend
uv run python tts_api.py \
  --mode cosyvoice3 \
  --voice jason \
  --text "这是一个本地 CosyVoice 三代旁白测试。" \
  --output output/cosyvoice3_test.wav

# MOSS-TTS Nano, recommended for Mac comparison
uv run python tts_api.py \
  --mode moss_nano \
  --voice official_female \
  --text "这是一个本地 MOSS TTS Nano 旁白测试。" \
  --output output/moss_nano_test.wav

# MOSS-TTS Local Transformer, slower 1.7B comparison
uv run python tts_api.py \
  --mode moss_local \
  --voice official_female \
  --text "这是一个本地 MOSS TTS Local Transformer 旁白测试。" \
  --output output/moss_local_test.wav
```

## 添加新声音

1. 录制一段 **3 秒左右**的清晰语音，保存为 wav 格式，放到 `voices/` 目录
2. 在 `demo.py` 的 `VOICE_PROFILES` 中添加一条配置：

```python
VOICE_PROFILES = {
    "jason": {
        "ref_audio": str(VOICES_DIR / "jason.wav"),
        "ref_text": "大家好,我是Jason.欢迎回到我的频道.今天给大家讲一段Tesla的故事",
    },
    "alice": {
        "ref_audio": str(VOICES_DIR / "alice.wav"),
        "ref_text": "transcript of the reference audio",
    },
}
```

3. 运行：`uv run python demo.py base alice`

> **注意**：`ref_text` 必须是参考音频中**实际说的内容**的准确文字转录，否则克隆效果会变差。

## 注意事项

1. **`mlx-audio` 版本**：必须 >= 0.3.0，旧版本（0.2.x）没有 `qwen3_tts` 模块，会报 `ModelConfig.__init__() missing required positional arguments` 错误
2. **Pre-release 依赖**：`mlx-audio` 0.3.0 依赖 `transformers==5.0.0rc3`（预发布版），`pyproject.toml` 中已配置 `[tool.uv] prerelease = "allow"`
3. **内存占用**：模型推理峰值约 6 GB，确保 Mac 有足够的统一内存
4. **运行时警告**：`tokenizer incorrect regex pattern` 和 `model type mismatch` 的警告可以忽略，不影响生成结果

## 项目结构

```
jk_tts/
├── voices/
│   └── jason.wav        # 参考音频（用于声音克隆）
├── demo.py              # TTS 生成示例（支持 base / voice_design 切换）
├── main.py              # 入口（占位）
├── pyproject.toml       # 项目配置与依赖
├── uv.lock              # 依赖锁定文件
└── README.md
```
