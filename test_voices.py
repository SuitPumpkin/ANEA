"""Test multi-voz: recorre todas las voces y cambia entre ellas."""

import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anea import config
from anea.tts import TTSEngine

print("=== Init TTS engine ===")
tts = TTSEngine()
print(f"  available voices: {tts.available_voices}")
print(f"  labels: {[tts.voice_labels[v] for v in tts.available_voices]}")
print(f"  default: {tts.voice}")

for voice in tts.available_voices:
    label = tts.voice_labels[voice]
    print(f"\n=== Warmup {label} ({voice}) ===")
    t0 = time.time()
    tts.warmup(voice=voice)
    print(f"  warmup done in {time.time() - t0:.1f}s")

    tts.set_voice(voice)
    text = f"Hola, soy {label}, tu asistente de voz."
    t0 = time.time()
    wav = tts.synthesize(text)
    dt = time.time() - t0
    dur = wav.size / config.TTS_SAMPLE_RATE
    backend = "Silero" if tts._voices[voice].using_fallback else "Kokoro"
    print(f"  {dt:.2f}s, {wav.size} samples ({dur:.2f}s audio, rtf={dt / max(dur, 1e-9):.3f}) backend={backend}")
    sf.write(str(Path(f"temp/voice_test_{voice}.wav")), wav, config.TTS_SAMPLE_RATE)

print(f"\n=== Vuelta a {tts.voice_labels[tts.available_voices[0]]} ===")
tts.set_voice(tts.available_voices[0])
t0 = time.time()
tts.synthesize("Volviendo a la voz principal.")
print(f"  {time.time() - t0:.2f}s")
print("OK")
