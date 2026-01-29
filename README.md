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
