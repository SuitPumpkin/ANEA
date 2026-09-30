"""Test TTS end-to-end con Kokoro y guarda el WAV."""

import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea import config
from anea.fasttts import kokoro_available
from anea.tts import TTSEngine

tts = TTSEngine()
print("Backend Kokoro instalado:", kokoro_available())
print(f"Loading TTS model (voices: {tts.available_voices})...")
t0 = time.time()
tts.initialize()
print(f"Model loaded in {time.time() - t0:.1f}s")

text = "Hola, esto es una prueba de la sintesis de voz sin clonacion."
print(f"Synthesizing {len(text)} chars...")
t0 = time.time()
wav = tts.synthesize(text)
dt = time.time() - t0
dur = wav.size / config.TTS_SAMPLE_RATE
print(f"Synthesized in {dt:.2f}s, length={wav.size} samples ({dur:.2f}s, rtf={dt / max(dur, 1e-9):.3f})")

out_path = Path("temp/test_tts_output.wav")
out_path.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out_path), wav, config.TTS_SAMPLE_RATE)
print(f"Saved to {out_path}")
print("OK")
