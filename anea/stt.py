"""Speech-to-Text con faster-whisper."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

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
        sr = sample_rate or config.SAMPLE_RATE_INPUT
        tmp_path: Path | None = None
        try:
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = Path(f.name)
            sf.write(str(tmp_path), audio.astype(np.float32), sr)
            segments, info = self._model.transcribe(
                str(tmp_path),
                language=self.language,
                beam_size=5,
                vad_filter=True,
                initial_prompt=self.initial_prompt,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            lang = getattr(info, "language", None) or self.language
            log.info("STT (%s): %s", lang, text)
            return TranscriptionResult(text=text, language=lang)
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except Exception:
                    pass