"""Test loading Mistral 7B GGUF with GPU offload and a quick generation."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from llama_cpp import Llama

model_path = r"D:\Mis Programas\ANEA\Models\mistral-7b-instruct-v0.2.Q4_K_M.gguf"
print(f"Loading {Path(model_path).name}...")
t0 = time.time()
llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False,
)
print(f"Loaded in {time.time() - t0:.1f}s")

messages = [
    {"role": "user", "content": "Hola, ¿cómo estás? Responde en español, en 1 frase corta."},
]
print("Generating...")
t0 = time.time()
result = llm.create_chat_completion(
    messages=messages,
    temperature=0.5,
    max_tokens=80,
    repeat_penalty=1.15,
    top_p=0.9,
    top_k=40,
)
text = result["choices"][0]["message"]["content"].strip()
print(f"Generated in {time.time() - t0:.1f}s:")
print(f"  >>> {text}")