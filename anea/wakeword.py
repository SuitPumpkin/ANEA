"""Detector de wake word basado en openWakeWord."""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

from . import config

log = logging.getLogger(__name__)


class WakeWordDetector:
    """Wrapper sobre openWakeWord para correr en hilo daemon."""

    def __init__(
        self,
        on_detect: Callable[[str], None],
        model_name: str = config.WAKEWORD_MODEL_NAME,
        threshold: float = config.WAKEWORD_THRESHOLD,
        sample_rate: int = config.WAKEWORD_SAMPLE_RATE,
        chunk_size: int = 1280,
    ) -> None:
        self._on_detect = on_detect
        self.model_name = model_name
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._model = None
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._stream = None
        self._chunk_count = 0

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def _load_model(self) -> None:
        if not config.WAKEWORD_ENABLED:
            log.info("Wake word deshabilitado por configuración (WAKEWORD_ENABLED=False)")
            return
        try:
            from openwakeword.model import Model  # type: ignore

            log.info("Cargando modelos openWakeWord (todos los pre-entrenados)")
            self._model = Model(
                wakeword_models=[],
                inference_framework="tflite",
            )
            log.info(
                "Modelos openWakeWord cargados: %s (threshold=%.2f)",
                list(self._model.models.keys()),
                self.threshold,
            )
        except Exception as exc:
            log.warning(
                "No se pudo cargar openWakeWord: %s. "
                "Wake word deshabilitado; usa activación manual.",
                exc,
            )
            self._model = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._load_model()
        if self._model is None:
            return
        self._running.set()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="wakeword")
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        import sounddevice as sd

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=config.WAKEWORD_CHANNELS,
                dtype="int16",
                blocksize=self.chunk_size,
            )
            self._stream.start()
        except Exception as exc:
            log.error("No se pudo abrir stream de wake word: %s", exc)
            self._running.clear()
            return

        log.info("Wake word listener activo")
        while self._running.is_set():
            try:
                data, _ = self._stream.read(self.chunk_size)
            except Exception as exc:
                log.error("Error leyendo audio wake word: %s", exc)
                break
            audio = data.flatten()
            try:
                prediction = self._model.predict(audio)
            except Exception as exc:
                log.error("Error en predicción wake word: %s", exc)
                continue
            self._chunk_count += 1
            for name, scores in prediction.items():
                score = float(np.max(scores)) if hasattr(scores, "__len__") else float(scores)
                if score >= self.threshold:
                    log.info("Wake word detectado: %s (score=%.3f)", name, score)
                    try:
                        self._on_detect(name)
                    except Exception as exc:
                        log.exception("Error en callback de wake word: %s", exc)
                    while self._running.is_set():
                        try:
                            self._stream.read(self.chunk_size)
                        except Exception:
                            break
                    break