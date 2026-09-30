"""TTS multi-voz para ANEA.

Voces sintéticas pre-entrenadas: Kokoro-82M como backend principal y Silero
como respaldo si Kokoro no está disponible. No hay clonación de voz: las
muestras de ``Voice Samples/`` no se leen en tiempo de ejecución.
"""

from __future__ import annotations

import logging
import re
import tempfile
import threading
from pathlib import Path

import numpy as np

from . import config
from .fasttts import FastVoice, kokoro_available

log = logging.getLogger(__name__)


def _split_for_tts(text: str, max_chars: int) -> list[str]:
    """Divide respuestas largas en unidades estables para el sintetizador."""
    normalized = re.sub(r"\s+", " ", (text or "").strip())
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?¿¡])\s+", normalized)
    chunks: list[str] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        while len(sentence) > max_chars:
            split_at = sentence.rfind(" ", 0, max_chars)
            if split_at < max_chars // 2:
                split_at = max_chars
            chunks.append(sentence[:split_at].strip())
            sentence = sentence[split_at:].strip()
        if sentence:
            chunks.append(sentence)
    return chunks


class SileroVoice:
    """Backend de respaldo: Silero v3 (rápido, sin red, voz sintética)."""

    def __init__(
        self,
        voice_id: str = "silero",
        language: str = config.SILERO_LANGUAGE,
        speaker: str = config.SILERO_SPEAKER,
        sample_rate: int = config.TTS_SAMPLE_RATE,
    ) -> None:
        self.voice_id = voice_id
        self.language = language
        self.speaker = speaker
        self._sample_rate = sample_rate
        self._tts = None
        self._init_lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def initialize(self) -> None:
        if self._tts is not None:
            return
        with self._init_lock:
            if self._tts is not None:
                return
            from silero_tts.silero_tts import SileroTTS  # type: ignore

            log.info("Cargando Silero v3_%s (speaker=%s)", self.language, self.speaker)
            self._tts = SileroTTS(
                model_id=f"v3_{self.language}",
                language=self.language,
                speaker=self.speaker,
                sample_rate=self._sample_rate,
            )
            log.info("Silero listo (sr=%d, speaker=%s)", self._sample_rate, self.speaker)

    def synthesize(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(1, dtype=np.float32)
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = Path(f.name)
        try:
            self._tts.tts(text, str(tmp_path))
            audio, _ = sf.read(str(tmp_path), dtype="float32")
        finally:
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return np.asarray(audio, dtype=np.float32)


class _VoiceWithFallback:
    """Kokoro con degradación silenciosa a Silero si el backend falla.

    El primer fallo de inicialización o de síntesis cambia a Silero para el
    resto de la sesión y lo deja registrado en el log.
    """

    def __init__(self, voice_id: str, primary: FastVoice, fallback: SileroVoice) -> None:
        self.voice_id = voice_id
        self.primary = primary
        self.fallback = fallback
        self.using_fallback = False
        self._lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self.primary.sample_rate

    def initialize(self) -> None:
        if not self.using_fallback:
            try:
                self.primary.initialize()
                return
            except Exception as exc:
                with self._lock:
                    if not self.using_fallback:
                        self.using_fallback = True
                        log.warning(
                            "Kokoro no disponible (%s); la voz '%s' pasa a Silero",
                            exc,
                            self.voice_id,
                        )
        self.fallback.initialize()

    def _active(self):
        return self.fallback if self.using_fallback else self.primary

    def synthesize(self, text: str) -> np.ndarray:
        try:
            return self._active().synthesize(text)
        except Exception as exc:
            if self.using_fallback:
                raise
            with self._lock:
                already = self.using_fallback
                self.using_fallback = True
            if already:
                raise
            log.warning(
                "Fallo de Kokoro en la voz '%s' (%s); cambiando a Silero",
                self.voice_id,
                exc,
            )
            self.fallback.initialize()
            return self.fallback.synthesize(text)


class TTSEngine:
    """Engine multi-voz. Cambia voz al instante con set_voice()."""

    def __init__(
        self,
        voice: str | None = None,
        language: str = config.TTS_LANGUAGE,
    ) -> None:
        self.language = language
        self._voices: dict[str, _VoiceWithFallback] = {}
        self._voice_order: list[str] = []
        requested = voice or config.DEFAULT_VOICE
        self._init_voices(default=requested)
        self._voice = self._resolve_default(requested)

    def _init_voices(self, default: str) -> None:
        if not kokoro_available():
            log.warning(
                "Kokoro no está instalado; todas las voces usarán Silero. "
                "Instálalo con: pip install kokoro soundfile"
            )
        for voice_id, spec in config.TTS_VOICES.items():
            self._voices[voice_id] = _VoiceWithFallback(
                voice_id=voice_id,
                primary=FastVoice(voice_id, spec["kokoro"]),
                fallback=SileroVoice(voice_id=voice_id),
            )
            self._voice_order.append(voice_id)
        if default and default not in self._voices:
            log.warning("Voz '%s' desconocida; se usa la primera disponible", default)

    def _resolve_default(self, default: str) -> str:
        if default in self._voices:
            return default
        return self._voice_order[0] if self._voice_order else config.DEFAULT_VOICE

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def available_voices(self) -> list[str]:
        return list(self._voice_order)

    @property
    def voice_labels(self) -> dict[str, str]:
        return config.voice_labels()

    def set_voice(self, name: str) -> None:
        name = name.lower()
        if name in self._voices and name != self._voice:
            log.info("Cambiando voz a '%s'", name)
            self._voice = name

    def initialize(self, voice: str | None = None) -> None:
        if voice is not None and voice != self._voice:
            self.set_voice(voice)
        self._voices[self._voice].initialize()

    def warmup(self, voice: str | None = None) -> None:
        v_name = voice or self._voice
        v = self._voices[v_name]
        v.initialize()
        try:
            audio = v.synthesize("Hola.")
            log.info("Warmup '%s' listo: %d samples", v_name, len(audio))
        except Exception as exc:
            log.warning("Warmup '%s' falló: %s", v_name, exc)

    def synthesize(self, text: str) -> np.ndarray:
        rendered = list(self.synthesize_chunks(text))
        if not rendered:
            return np.zeros(1, dtype=np.float32)
        return np.concatenate(rendered)

    def synthesize_chunks(self, text: str):
        """Genera fragmentos reproducibles para empezar a hablar cuanto antes."""
        chunks = _split_for_tts(text, config.TTS_CHUNK_MAX_CHARS)
        if not chunks:
            return
        v = self._voices[self._voice]
        v.initialize()
        pause = np.zeros(
            int(config.TTS_SAMPLE_RATE * config.TTS_CHUNK_PAUSE_SECONDS),
            dtype=np.float32,
        )
        for index, chunk in enumerate(chunks):
            yield v.synthesize(chunk)
            if index < len(chunks) - 1:
                yield pause

    def current_persona(self, voice: str | None = None) -> str:
        return persona_for_voice(voice or self._voice)


_VOICE_FORMAT_RULES = (
    "FORMATO DE RESPUESTA:\n"
    "- Escribe de 1 a 3 frases conectadas con naturalidad, entre 15 y 80 palabras.\n"
    "- Termina siempre en punto, signo de interrogación o exclamación.\n"
    "- PROHIBIDO: listas, viñetas, markdown (**negrita**, # títulos, -, 1.), saltos de línea, "
    "párrafos separados, código, URLs.\n"
    "- PROHIBIDO: emojis o símbolos tipo 🙂, 😅, 🎉, ❤️. Usa 'jajaja', 'jeje', 'uy', 'uf', "
    "'vaya', 'qué bien', 'ay por favor', 'dale', 'venga'.\n"
    "- Solo caracteres ASCII imprimibles más letras acentuadas (áéíóúñü) y signos ¿¡.,;:!?.\n"
    "- Si no sabes algo, admítelo. No inventes datos concretos (fechas, cifras) que no te hayan dado.\n"
)

_LANGUAGE_RULE = (
    "IMPORTANT: You must reply ONLY in Spanish. Never use English, never add translations "
    "in parentheses, never use words from other languages. Write pure Spanish text only.\n\n"
)

VOICE_PERSONAS = {
    "dora": (
        _LANGUAGE_RULE
        + "Eres Anea, una asistente personal de voz llamada Anea. Hablas en español "
        "de forma natural, cálida y directa, como una amiga cercana.\n\n"
        + _VOICE_FORMAT_RULES
        + "- Si el usuario te saluda, salúdalo y pregunta en qué puede ayudar."
    ),
    "maria": (
        _LANGUAGE_RULE
        + "Eres María, una asistente personal de voz llamada María. Hablas en español "
        "de forma tranquila, serena y atenta.\n\n"
        + _VOICE_FORMAT_RULES
        + "- Si el usuario te saluda, salúdalo y pregunta en qué puede ayudar."
    ),
}


def persona_for_voice(voice: str) -> str:
    return VOICE_PERSONAS.get(voice.lower(), VOICE_PERSONAS[config.DEFAULT_VOICE])

