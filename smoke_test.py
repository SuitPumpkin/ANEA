"""Smoke test: instancia todos los componentes sin abrir el mainloop.

Falla rápido si falta algún módulo o ruta.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea import audio, config, llm, stt, tts, wakeword  # noqa: F401
from anea.fasttts import kokoro_available
from anea.gui import ANEAGui  # noqa: F401


def main() -> int:
    config.ensure_dirs()
    print("Project root:", config.PROJECT_ROOT)
    print("Kokoro instalado:", kokoro_available())
    print("Voces:", {v: config.TTS_VOICES[v]["label"] for v in config.TTS_VOICES})
    print("Voz por defecto:", config.resolve_user_default_voice())
    print("Wake word habilitado:", config.WAKEWORD_ENABLED, "->", config.WAKEWORD_MODEL_NAME)
    for key, p in config.EMOTIONS.items():
        print(f"  emotion[{key}]: exists={p.exists()}")
    print("GGUF:", config.GGUF_MODEL_PATH.exists(), config.GGUF_MODEL_PATH.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
