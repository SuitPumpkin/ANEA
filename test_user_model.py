"""Test loading the user-selected Gemma GGUF."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llama_cpp import Llama

model_path = r"D:\Mis Programas\ANEA\Models\Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_Q8_0.gguf"
print(f"Loading {Path(model_path).name}...")
t0 = time.time()
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=8,
        n_gpu_layers=-1,
        chat_format="gemma",
        verbose=True,
    )
    print(f"Loaded in {time.time() - t0:.1f}s")
    print("Testing generation...")
    result = llm.create_chat_completion(
        messages=[{"role": "user", "content": "Di hola en español."}],
        max_tokens=40,
        temperature=0.5,
    )
    print(">>>", result["choices"][0]["message"]["content"])
except Exception as exc:
    import traceback
    traceback.print_exc()
    print(f"FAILED after {time.time() - t0:.1f}s: {type(exc).__name__}: {exc}")