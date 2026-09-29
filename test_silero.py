"""Test Silero TTS with each Spanish speaker."""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from silero_tts.silero_tts import SileroTTS

for speaker in ["es_0", "es_2"]:
    print(f"\n=== speaker: {speaker} ===")
    tts = SileroTTS(model_id="v3_es", language="es", speaker=speaker, sample_rate=24000)
    for i in range(3):
        t0 = time.time()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        text = "Hola, soy una prueba de voz rapida." if i == 0 else f"Segunda llamada numero {i + 1}."
        tts.tts(text, tmp_path)
        audio, sr = sf.read(tmp_path, dtype="float32")
        print(f"  call {i}: {time.time() - t0:.2f}s ({len(audio)/sr:.2f}s audio)")
    out = Path(f"temp/silero_{speaker}.wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), audio, sr)
    print(f"Saved to {out}")