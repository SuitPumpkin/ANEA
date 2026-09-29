"""Speech-to-Text con faster-whisper."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from . import config

log = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    language: str | None


class STTEngine:
    """Wrapper ligero sobre faster-whisper."""

    def __init__(
        self,
        model_size: str = config.WHISPER_MODEL_SIZE,
        device: str = config.WHISPER_DEVICE,
        compute_type: str = config.WHISPER_COMPUTE_TYPE,
        language: str | None = config.WHISPER_LANGUAGE,
        initial_prompt: str | None = getattr(config, "WHISPER_INITIAL_PROMPT", None),
    ) -> None:
        self.model_size = model_size
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.initial_prompt = initial_prompt
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel  # type: ignore

        log.info(
            "Cargando faster-whisper '%s' (device=%s, compute=%s)",
            self.model_size,
            self.device,
            self.compute_type,
        )
        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        log.info("faster-whisper listo")

    def transcribe(self, audio: np.ndarray, sample_rate: int | None = None) -> TranscriptionResult:
        self._ensure_model()
        if audio.size == 0:
            return TranscriptionResult(text="", language=self.language)
        segments, info = self._model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language=self.language,
            beam_size=config.WHISPER_BEAM_SIZE,
            best_of=config.WHISPER_BEST_OF,
            vad_filter=config.WHISPER_VAD_FILTER,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.initial_prompt,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        lang = getattr(info, "language", None) or self.language
        log.info("STT (%s): %s", lang, text)
        return TranscriptionResult(text=text, language=lang)