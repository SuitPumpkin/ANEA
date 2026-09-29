"""Test Mistral with the new prompt to verify it stays in Spanish."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea.llm import LLMEngine

llm = LLMEngine()
llm._ensure_model()

prompts = [
    "Hola, ¿cómo estás?",
    "¿Qué puedes hacer?",
    "Dime un dato curioso",
    "¿Quién eres?",
    "Tengo 25 años, ¿qué me recomiendas?",
]

for p in prompts:
    print(f"\n>>> {p}")
    r = llm.generate(p)
    print(f"<<< {r}")