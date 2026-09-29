"""Smoke test: instantiate all components without running mainloop.

Falla rápido si falta algún módulo o ruta.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea import audio, config, llm, stt, tts, wakeword  # noqa: F401
from anea.gui import ANEAGui  # noqa: F401


def main() -> int:
    config.ensure_dirs()
    print("Project root:", config.PROJECT_ROOT)
    print("Voice sample:", config.discover_voice_sample())
    for key, p in config.EMOTIONS.items():
        print(f"  emotion[{key}]: exists={p.exists()}")
    print("GGUF:", config.GGUF_MODEL_PATH.exists(), config.GGUF_MODEL_PATH.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())