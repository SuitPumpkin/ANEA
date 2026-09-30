"""TTS rápido para ANEA sobre Kokoro-82M.

Voces pre-entrenadas de 24 kHz, síntesis en el propio proceso y sin
clonación: no se usa ninguna muestra de voz del usuario ni se guardan
embeddings. El modelo se comparte entre todas las instancias de voz para no
duplicar los ~330 MB de pesos en memoria.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from . import config

log = logging.getLogger(__name__)

_model_lock = threading.Lock()
_shared_model = None


def _get_shared_model():
    """Carga (una sola vez) el modelo Kokoro compartido por todas las voces."""
    global _shared_model
    if _shared_model is not None:
        return _shared_model
    with _model_lock:
        if _shared_model is not None:
            return _shared_model
        import torch
        from kokoro import KModel

        device = "cpu"
        if config.KOKORO_USE_GPU and torch.cuda.is_available():
            device = "cuda"
        log.info("Cargando Kokoro-82M (%s) desde %s...", device, config.KOKORO_REPO_ID)
        model = KModel(repo_id=config.KOKORO_REPO_ID).eval().to(device)
        _shared_model = model
        log.info("Kokoro-82M listo en %s", device)
    return _shared_model


def kokoro_available() -> bool:
    """True si el paquete kokoro está instalado y el modelo se puede cargar."""
    try:
        import kokoro  # noqa: F401
    except Exception:
        return False
    return True


class FastVoice:
    """Una voz Kokoro. Misma interfaz que el resto de backends de TTS."""

    def __init__(
        self,
        voice_id: str,
        kokoro_voice: str,
        lang_code: str = config.KOKORO_LANG_CODE,
        speed: float = config.KOKORO_SPEED,
        sample_rate: int = config.TTS_SAMPLE_RATE,
    ) -> None:
        self.voice_id = voice_id
        self.kokoro_voice = kokoro_voice
        self.lang_code = lang_code
        self.speed = speed
        self._sample_rate = sample_rate
        self._pipeline = None
        self._device = "cpu"
        self._init_lock = threading.Lock()
        # KPipeline es un generador: no admite dos llamadas concurrentes.
        self._synth_lock = threading.Lock()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def initialize(self) -> None:
        if self._pipeline is not None:
            return
        with self._init_lock:
            if self._pipeline is not None:
                return
            from kokoro import KPipeline

            model = _get_shared_model()
            try:
                import torch

                self._device = str(next(model.parameters()).device)
            except Exception:
                self._device = "cpu"
            self._pipeline = KPipeline(
                lang_code=self.lang_code,
                model=model,
                repo_id=config.KOKORO_REPO_ID,
                device=self._device,
            )
            log.info("Voz '%s' lista (kokoro=%s, device=%s)", self.voice_id, self.kokoro_voice, self._device)

    def synthesize(self, text: str) -> np.ndarray:
        text = (text or "").strip()
        if not text:
            return np.zeros(1, dtype=np.float32)
        self.initialize()
        with self._synth_lock:
            parts = [
                np.asarray(audio, dtype=np.float32)
                for _, _, audio in self._pipeline(
                    text, voice=self.kokoro_voice, speed=self.speed
                )
            ]
        if not parts:
            return np.zeros(1, dtype=np.float32)
        audio = np.concatenate(parts)
        return np.clip(audio, -1.0, 1.0)
