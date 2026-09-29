"""Test TTS end-to-end and save WAV."""

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea import config
from anea.tts import TTSEngine

tts = TTSEngine(voice_sample_path=config.discover_voice_sample())
print("Loading TTS model...")
t0 = time.time()
tts.initialize()
print(f"Model loaded in {time.time() - t0:.1f}s")
print("Synthesizing 'Hola, esto es una prueba'...")
t0 = time.time()
wav = tts.synthesize("Hola, esto es una prueba.")
print(f"Synthesized in {time.time() - t0:.1f}s, length={wav.size} samples ({wav.size / 24000:.2f}s)")

out_path = Path("temp/test_tts_output.wav")
out_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out_path), wav, 24000)
print(f"Saved to {out_path}")
print("OK")