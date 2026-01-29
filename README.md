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

## 使用

```bash
uv run python demo.py
```

生成的音频会保存为 `test_audio_000.wav`。

### Python API

```python
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.generate import generate_audio

model = load_model("mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16")
generate_audio(
    model=model,
    text="你好，这是一段测试语音。",
    instruct="A cheerful young female voice with clear pronunciation and moderate speed",
    file_prefix="output",
)
```

### CLI

```bash
uv run python -m mlx_audio.tts.generate \
  --model mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16 \
  --text "Hello, this is a test." \
  --instruct "A calm male voice with low pitch"
```

## 注意事项

1. **必须使用 `instruct` 参数**：VoiceDesign 模型通过文字描述来控制声音风格（如音色、语速、情感），不接受 `ref_audio` 参考音频
2. **`mlx-audio` 版本**：必须 >= 0.3.0，旧版本（0.2.x）没有 `qwen3_tts` 模块，会报 `ModelConfig.__init__() missing required positional arguments` 错误
3. **Pre-release 依赖**：`mlx-audio` 0.3.0 依赖 `transformers==5.0.0rc3`（预发布版），`pyproject.toml` 中已配置 `[tool.uv] prerelease = "allow"`
4. **内存占用**：模型推理峰值约 6 GB，确保 Mac 有足够的统一内存
5. **运行时警告**：`tokenizer incorrect regex pattern` 和 `model type mismatch` 的警告可以忽略，不影响生成结果

## 项目结构

```
jk_tts/
├── demo.py          # TTS 生成示例
├── main.py          # 入口（占位）
├── pyproject.toml   # 项目配置与依赖
├── uv.lock          # 依赖锁定文件
└── README.md
```
