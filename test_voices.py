"""Test multi-voice TTS."""

import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea import config
from anea.tts import TTSEngine

print("=== Init TTS engine (discovers voices) ===")
tts = TTSEngine(voice="anea")
print(f"  available voices: {tts.available_voices}")
print(f"  default: {tts.voice}")

print("=== Warmup Anea (XTTS) ===")
t0 = time.time()
tts.warmup(voice="anea")
print(f"  warmup done in {time.time() - t0:.1f}s")

print("\n=== Warmup Miku (Silero) ===")
t0 = time.time()
tts.warmup(voice="miku")
print(f"  warmup done in {time.time() - t0:.1f}s")

tts.set_voice("alex")
print("\n=== Alex synthesis ===")
t0 = time.time()
wav = tts.synthesize("Hola, soy Alex, con una voz distinta.")
print(f"  {time.time() - t0:.2f}s, {len(wav)} samples ({len(wav)/24000:.2f}s audio)")
sf.write(str(Path("temp/voice_test_alex.wav")), wav, 24000)

tts.set_voice("anea")
print("\n=== Anea synthesis ===")
t0 = time.time()
wav = tts.synthesize("Hola, soy Anea, tu asistente de voz.")
print(f"  {time.time() - t0:.2f}s, {len(wav)} samples ({len(wav)/24000:.2f}s audio)")
out = Path("temp/voice_test_anea.wav")
out.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out), wav, 24000)

tts.set_voice("miku")
print("\n=== Miku synthesis ===")
t0 = time.time()
wav = tts.synthesize("Hola, soy Miku, tu asistente super rapida.")
print(f"  {time.time() - t0:.2f}s, {len(wav)} samples ({len(wav)/24000:.2f}s audio)")
sf.write(str(Path("temp/voice_test_miku.wav")), wav, 24000)

print("\n=== Switch back to Anea ===")
tts.set_voice("anea")
t0 = time.time()
wav = tts.synthesize("Volviendo a Anea.")
print(f"  {time.time() - t0:.2f}s")
print("OK")