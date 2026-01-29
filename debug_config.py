
import json
from pathlib import Path
from mlx_lm.models.qwen2 import ModelArgs as Qwen2ModelArgs
# Note: mlx_audio imports qwen3 from mlx_lm, but let's check if it exists or maps to qwen2.
try:
    from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelArgs
    print("Imported Qwen3ModelArgs from mlx_lm")
except ImportError:
    print("Could not import qwen3 from mlx_lm, trying qwen2")
    Qwen3ModelArgs = Qwen2ModelArgs

from mlx_audio.tts.utils import load_config

model_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"

print(f"Loading config for {model_id}...")
try:
    config = load_config(model_id)
    print("Config keys:", config.keys())
    print("\nFull Config:")
    print(json.dumps(config, indent=2))
    
    # Try to instantiate ModelArgs
    print("\nAttempting to instantiate ModelArgs from config...")
    try:
        args = Qwen3ModelArgs.from_dict(config)
        print("Successfully instantiated ModelArgs")
    except TypeError as e:
        print(f"Failed to instantiate ModelArgs: {e}")
        # Inspect what keys are expected vs present
        import inspect
        sig = inspect.signature(Qwen3ModelArgs.__init__)
        print("\nModelArgs.__init__ parameters:")
        for name, param in sig.parameters.items():
            if name != 'self':
                has_default = param.default != inspect.Parameter.empty
                in_config = name in config
                print(f"  {name}: default={has_default}, in_config={in_config}")

except Exception as e:
    print(f"Error loading config: {e}")
