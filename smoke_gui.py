"""Smoke test: build GUI and progress through loading to ready state."""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea.gui import ANEAGui
from anea import config


def main() -> int:
    gui = ANEAGui()

    def init_steps():
        time.sleep(0.5)
        gui.set_loading("Inicializando micrófono... OK", ["micrófono: OK"])
        time.sleep(0.3)
        gui.set_loading("Inicializando STT... OK", ["micrófono: OK", "STT: OK"])
        time.sleep(0.3)
        gui.set_loading("Inicializando TTS... OK", ["micrófono: OK", "STT: OK", "TTS: OK"])
        time.sleep(0.3)
        gui.set_ready()

    threading.Thread(target=init_steps, daemon=True).start()
    gui.root.after(4000, gui.root.destroy)
    gui.mainloop()
    print("GUI ready OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())