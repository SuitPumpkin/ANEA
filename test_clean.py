"""Test LLM cleaning with sample English-mixed outputs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea.main import ANEA


tests = [
    "Hola, estoy bien gracias. (Hello, I'm fine, thank you.)",
    "Claro! (Of course!) ¿En qué te ayudo?",
    "Buenas, hoy hace sol. (It's sunny today.)",
    "¡Hola! *Sonríe* ¿Cómo estás?",
    "**Listo**. Aquí tienes la info.",
    "Una frase normal sin nada raro.",
    "Hola.\n(English translation here.)\nAdiós.",
    "Esto es un ejemplo (note: a side comment).",
]

complete_joke = "Aquí tienes un chiste:\n\n¿Qué le dice una impresora a otra?\n... ¡Imprime!"
assert ANEA._clean_reply(complete_joke) == (
    "Aquí tienes un chiste: ¿Qué le dice una impresora a otra? ... ¡Imprime!"
)

for t in tests:
    print(f"IN:  {t}")
    print(f"OUT: {ANEA._clean_reply(t)}\n")