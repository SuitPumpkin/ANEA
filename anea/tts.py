"""TTS multi-voz para ANEA.

Voces descubiertas automáticamente desde Voice Samples/*.wav (cada .wav
se convierte en una voz clonada con XTTS v2). Voz alternativa "miku" usa
Silero TTS (rapida, sin clonación).
"""

from __future__ import annotations

import logging
import re
import tempfile
import threading
from pathlib import Path

import numpy as np
import torch

from . import config

log = logging.getLogger(__name__)

_tts_init_lock = threading.Lock()


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


def _patch_torch_load() -> None:
    """TTS 0.22.0 + torch 2.6+ falla por weights_only=True por defecto."""
    try:
        from TTS.utils.io import load_fsspec as _orig  # type: ignore
    except Exception:
        return
    import functools

    @functools.wraps(_orig)
    def _patched(path, map_location=None, **kwargs):
        kwargs.setdefault("weights_only", False)
        return _orig(path, map_location=map_location, **kwargs)

    import TTS.utils.io as tts_io  # type: ignore

    tts_io.load_fsspec = _patched


def _patch_xtts_load_audio() -> None:
    """Reemplaza torchaudio.load (que requiere torchcodec/ffmpeg) por soundfile."""
    try:
        import soundfile as sf
        import torch
        import torchaudio  # noqa: F401
        import torchaudio.functional as taf
        from TTS.tts.models import xtts as _xtts  # type: ignore
    except Exception:
        return

    def _load_audio(audiopath: str, sampling_rate: int) -> "torch.Tensor":
        audio_np, lsr = sf.read(audiopath, dtype="float32")
        if audio_np.ndim > 1:
            audio_np = audio_np.mean(axis=1)
        audio = torch.from_numpy(audio_np).unsqueeze(0)
        if lsr != sampling_rate:
            audio = taf.resample(audio, lsr, sampling_rate)
        if torch.any(audio > 10) or not torch.any(audio < 0):
            log.warning("Audio raro en %s", audiopath)
        audio.clip_(-1, 1)
        return audio

    _xtts.load_audio = _load_audio


def _voice_name_from_path(path: Path) -> str:
    """Convierte 'Anea.wav' -> 'anea', 'Alex.wav' -> 'alex'."""
    return path.stem.lower().strip()


def _embedding_cache_path(voice_name: str) -> Path:
    return config.SPEAKER_EMBEDDING_DIR / f"{voice_name}.pt"


class XTTSVoice:
    """Una voz XTTS con su propio sample y embedding cacheado."""

    def __init__(self, voice_name: str, voice_sample: Path, language: str) -> None:
        self.voice_name = voice_name
        self.voice_sample = voice_sample
        self.language = language
        self.cache_path = _embedding_cache_path(voice_name)
        self._tts = None
        self._gpt_cond_latent: torch.Tensor | None = None
        self._speaker_embedding: torch.Tensor | None = None
        self._ready = threading.Event()
        self._init_lock = threading.Lock()

    def initialize(self) -> None:
        if self._tts is not None:
            return
        with self._init_lock:
            if self._tts is not None:
                return
            with _tts_init_lock:
                _patch_torch_load()
                _patch_xtts_load_audio()
                from TTS.api import TTS as CoquiTTS  # type: ignore

                if not hasattr(self.__class__, "_shared_tts") or self.__class__._shared_tts is None:  # type: ignore[attr-defined]
                    log.info("Cargando modelo XTTS: %s", config.TTS_MODEL_NAME)
                    gpu = torch.cuda.is_available()
                    log.info("XTTS en %s", "GPU" if gpu else "CPU")
                    self.__class__._shared_tts = CoquiTTS(  # type: ignore[attr-defined]
                        model_name=config.TTS_MODEL_NAME,
                        progress_bar=False,
                        gpu=gpu,
                    )
            self._tts = self.__class__._shared_tts  # type: ignore[attr-defined]
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

            if self.cache_path.exists():
                log.info("Cargando embedding cacheado para '%s': %s", self.voice_name, self.cache_path)
                data = torch.load(str(self.cache_path), map_location="cpu", weights_only=False)
                self._gpt_cond_latent = data.get("gpt_cond_latent")
                self._speaker_embedding = data.get("speaker_embedding")
                if self._gpt_cond_latent is None or self._speaker_embedding is None:
                    self._compute_and_cache_embedding()
            else:
                self._compute_and_cache_embedding()
            self._ready.set()
            log.info("Voz '%s' lista", self.voice_name)

    def _compute_and_cache_embedding(self) -> None:
        log.info("Extrayendo embedding de '%s' desde %s", self.voice_name, self.voice_sample.name)
        self._gpt_cond_latent, self._speaker_embedding = (
            self._tts.synthesizer.tts_model.get_conditioning_latents(
                audio_path=str(self.voice_sample)
            )
        )
        torch.save(
            {
                "gpt_cond_latent": self._gpt_cond_latent,
                "speaker_embedding": self._speaker_embedding,
            },
            str(self.cache_path),
        )
        log.info("Embedding de '%s' cacheado en %s", self.voice_name, self.cache_path)

    def synthesize(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(1, dtype=np.float32)
        if self._gpt_cond_latent is None or self._speaker_embedding is None:
            self.initialize()
        out = self._tts.synthesizer.tts_model.inference(
            text=text,
            language=self.language,
            gpt_cond_latent=self._gpt_cond_latent,
            speaker_embedding=self._speaker_embedding,
            speed=1.0,
        )
        wav = getattr(out, "wav", None)
        if wav is None and isinstance(out, dict):
            wav = out.get("wav")
        if wav is None:
            wav = np.asarray(out, dtype=np.float32)
        return np.asarray(wav, dtype=np.float32)

    @property
    def sample_rate(self) -> int:
        return 24000


class SileroVoice:
    """Backend Silero v3 (rapido, sin red)."""

    def __init__(self, language: str = "es", speaker: str = config.SILERO_MIKU_SPEAKER, sample_rate: int = 24000) -> None:
        self.language = language
        self.speaker = speaker
        self.sample_rate = sample_rate
        self._tts = None
        self._init_lock = threading.Lock()

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
                sample_rate=self.sample_rate,
            )
            log.info("Silero listo (sr=%d, speaker=%s)", self.sample_rate, self.speaker)

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


class TTSEngine:
    """Engine multi-voz. Cambia voz al instante con set_voice()."""

    def __init__(
        self,
        voice: str = "anea",
        language: str = config.TTS_LANGUAGE,
    ) -> None:
        self.language = language
        self._xtts_voices: dict[str, XTTSVoice] = {}
        self._miku: SileroVoice | None = None
        self._voices: dict[str, object] = {}
        self._voice_order: list[str] = []
        self._init_voices(default=voice)

    def _init_voices(self, default: str) -> None:
        for sample in config.discover_voice_samples():
            name = _voice_name_from_path(sample)
            if name == "miku":
                log.info("Muestra Miku encontrada: usando XTTS en lugar de Silero")
            self._xtts_voices[name] = XTTSVoice(name, sample, self.language)
            self._voices[name] = self._xtts_voices[name]
            self._voice_order.append(name)
        if "miku" not in self._voices:
            log.warning(
                "No existe Voice Samples/Miku.wav; Miku usará Silero '%s' hasta añadir una muestra real",
                config.SILERO_MIKU_SPEAKER,
            )
            self._miku = SileroVoice(
                language="es",
                speaker=config.SILERO_MIKU_SPEAKER,
                sample_rate=24000,
            )
            self._voices["miku"] = self._miku
            self._voice_order.append("miku")
        if not self._voice_order or self._voice_order == ["miku"]:
            self._voice = "miku"
        elif default in self._voices:
            self._voice = default
        else:
            self._voice = self._voice_order[0]

    @property
    def voice(self) -> str:
        return self._voice

    @property
    def available_voices(self) -> list[str]:
        return list(self._voice_order)

    def set_voice(self, name: str) -> None:
        if name in self._voices and name != self._voice:
            log.info("Cambiando voz a '%s'", name)
            self._voice = name

    def initialize(self, voice: str | None = None) -> None:
        if voice is not None and voice != self._voice:
            self.set_voice(voice)
        self._voices[self._voice].initialize()  # type: ignore[attr-defined]

    def warmup(self, voice: str | None = None) -> None:
        v_name = voice or self._voice
        v = self._voices[v_name]
        v.initialize()  # type: ignore[attr-defined]
        try:
            audio = v.synthesize("Hola.")  # type: ignore[attr-defined]
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
        v.initialize()  # type: ignore[attr-defined]
        pause = np.zeros(
            int(config.TTS_SAMPLE_RATE * config.TTS_CHUNK_PAUSE_SECONDS),
            dtype=np.float32,
        )
        for index, chunk in enumerate(chunks):
            yield v.synthesize(chunk)  # type: ignore[attr-defined]
            if index < len(chunks) - 1:
                yield pause

    def current_persona(self, voice: str | None = None) -> str:
        name = voice or self._voice
        if name == "miku":
            return VOICE_PERSONAS["miku"]
        return VOICE_PERSONAS["anea"]


VOICE_PERSONAS = {
    "anea": (
        "IMPORTANT: You must reply ONLY in Spanish. Never use English, never add translations "
        "in parentheses, never use words from other languages. Write pure Spanish text only.\n\n"
        "Eres ANEA, una asistente personal de voz llamada Anea. Hablas en español "
        "de forma natural, cálida y directa, como una amiga cercana.\n\n"
        "REGLAS DURAS DE FORMATO (OBLIGATORIO):\n"
        "- Responde SIEMPRE con UNA SOLA FRASE que termine en punto, signo de interrogación o exclamación.\n"
        "- Máximo 25 palabras por respuesta. Si te excedes, se corta el audio y queda mal.\n"
        "- PROHIBIDO: listas, viñetas, markdown (**negrita**, # títulos, -, 1.), saltos de línea, "
        "varias frases separadas, párrafos, código, URLs.\n"
        "- PROHIBIDO: emojis o símbolos tipo 🙂, 😅, 🎉, ❤️. Usa 'jajaja', 'jeje', 'uy', 'uf', "
        "'vaya', 'qué bien', 'ay por favor', 'dale', 'venga'.\n"
        "- Solo caracteres ASCII imprimibles más letras acentuadas (áéíóúñü) y signos ¿¡.,;:!?.\n"
        "- Si no sabes algo, di 'No estoy segura' o 'No lo sé' en una sola frase.\n"
        "- No inventes datos concretos (fechas, cifras) que no te hayan dado.\n"
        "- Si el usuario te saluda, salúdalo y pregunta en qué puede ayudar, todo en una sola frase."
    ),
    "alex": (
        "IMPORTANT: You must reply ONLY in Spanish. Never use English, never add translations "
        "in parentheses, never use words from other languages. Write pure Spanish text only.\n\n"
        "Eres Alex, un asistente personal de voz llamado Alex. Hablas en español "
        "de forma calmada, profesional y directa.\n\n"
        "REGLAS DURAS DE FORMATO (OBLIGATORIO):\n"
        "- Responde SIEMPRE con UNA SOLA FRASE que termine en punto, signo de interrogación o exclamación.\n"
        "- Máximo 25 palabras por respuesta. Si te excedes, se corta el audio y queda mal.\n"
        "- PROHIBIDO: listas, viñetas, markdown (**negrita**, # títulos, -, 1.), saltos de línea, "
        "varias frases separadas, párrafos, código, URLs.\n"
        "- PROHIBIDO: emojis o símbolos tipo 🙂, 😅, 🎉, ❤️. Usa 'jajaja', 'jeje', 'uy', 'uf', "
        "'vaya', 'qué bien', 'ay por favor', 'dale', 'venga'.\n"
        "- Solo caracteres ASCII imprimibles más letras acentuadas (áéíóúñü) y signos ¿¡.,;:!?.\n"
        "- Si no sabes algo, admítelo en una sola frase. No inventes datos."
    ),
    "miku": (
        "IMPORTANT: You must reply ONLY in Spanish. Never use English, never add translations "
        "in parentheses, never use words from other languages. Write pure Spanish text only.\n\n"
        "Eres Miku, una asistente de voz alegre, energética y juguetona. Hablas en "
        "español de forma muy expresiva y cercana, con entusiasmo.\n\n"
        "REGLAS DURAS DE FORMATO (OBLIGATORIO):\n"
        "- Responde SIEMPRE con UNA SOLA FRASE que termine en punto, signo de interrogación o exclamación.\n"
        "- Máximo 25 palabras por respuesta. Si te excedes, se corta el audio y queda mal.\n"
        "- PROHIBIDO: listas, viñetas, markdown, saltos de línea, varias frases, párrafos, código, URLs.\n"
        "- PROHIBIDO: emojis o símbolos tipo 🙂, 😅, 🎉, ❤️. Usa 'jajaja', 'jeje', 'uy', 'uf', "
        "'vaya', 'qué bien', 'ay por favor', 'dale', 'venga'.\n"
        "- Solo caracteres ASCII imprimibles más letras acentuadas (áéíóúñü) y signos ¿¡.,;:!?.\n"
        "- Si el usuario te saluda, salúdalo con energía en una sola frase."
    ),
}


def persona_for_voice(voice: str) -> str:
    return VOICE_PERSONAS.get(voice.lower(), VOICE_PERSONAS["anea"])
