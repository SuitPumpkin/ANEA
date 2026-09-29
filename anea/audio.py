"""Manejo de audio: captura de micrófono, waveform, grabación y reproducción.

Usa streams separados de `sounddevice` para evitar conflictos de lock en
Windows (entrada vs salida en el mismo dispositivo).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import sounddevice as sd
import soundfile as sf

from . import config


class AudioError(RuntimeError):
    """Error general de audio."""


class AudioCapture:
    """Captura continua del micrófono con buffer circular para waveform."""

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE_INPUT,
        channels: int = config.CHANNELS,
        waveform_seconds: float = 2.0,
        device: int | str | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._waveform_len = int(waveform_seconds * sample_rate)
        self._buf: deque[np.ndarray] = deque()
        self._buf_lock = threading.Lock()
        self._stream: sd.InputStream | None = None
        self._record_buffer: list[np.ndarray] = []
        self._recording = False
        self._record_lock = threading.Lock()
        self._listeners: list[Callable[[np.ndarray], None]] = []
        self._device = device

    @property
    def is_active(self) -> bool:
        return self._stream is not None

    def set_device(self, device: int | str | None) -> None:
        was_active = self._stream is not None
        if was_active:
            self.stop()
        self._device = device
        if was_active:
            self.start()

    def start(self) -> None:
        if self._stream is not None:
            return
        kwargs = dict(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=config.DTYPE,
            blocksize=int(self.sample_rate * 0.05),
            callback=self._on_audio,
        )
        if self._device is not None:
            kwargs["device"] = self._device
        self._stream = sd.InputStream(**kwargs)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def add_listener(self, cb: Callable[[np.ndarray], None]) -> None:
        self._listeners.append(cb)

    def get_waveform(self) -> np.ndarray:
        with self._buf_lock:
            if not self._buf:
                return np.zeros(self._waveform_len, dtype=np.float32)
            data = np.concatenate(list(self._buf))[-self._waveform_len:]
        if data.size < self._waveform_len:
            pad = np.zeros(self._waveform_len - data.size, dtype=np.float32)
            data = np.concatenate([pad, data])
        return data

    def begin_recording(self) -> None:
        with self._record_lock:
            self._record_buffer = []
            self._recording = True

    def append_recording(self, chunk: np.ndarray) -> None:
        with self._record_lock:
            if self._recording:
                self._record_buffer.append(chunk.copy())

    def end_recording(self) -> np.ndarray:
        with self._record_lock:
            self._recording = False
            if not self._record_buffer:
                return np.zeros(1, dtype=np.float32)
            return np.concatenate(self._record_buffer)

    def save_wav(self, data: np.ndarray, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), data, self.sample_rate)
        return path

    def _on_audio(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            return
        chunk = indata[:, 0].astype(np.float32, copy=True)
        with self._buf_lock:
            self._buf.append(chunk)
            total = sum(len(c) for c in self._buf)
            while total > self._waveform_len and len(self._buf) > 1:
                self._buf.popleft()
                total = sum(len(c) for c in self._buf)
        if self._recording:
            self.append_recording(chunk)
        for cb in list(self._listeners):
            try:
                cb(chunk)
            except Exception:
                pass


def play_audio(data: np.ndarray, sample_rate: int = config.SAMPLE_RATE_OUTPUT) -> None:
    """Reproduce audio en stream de salida dedicado."""
    if data.size == 0:
        return
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    sd.play(data, samplerate=sample_rate, blocking=True)


def play_audio_stream(
    chunks: Iterable[np.ndarray],
    sample_rate: int = config.SAMPLE_RATE_OUTPUT,
    on_first_chunk: Callable[[np.ndarray], None] | None = None,
) -> None:
    """Reproduce chunks sin cerrar el dispositivo entre fragmentos."""
    first = True
    with sd.OutputStream(
        samplerate=sample_rate, channels=1, dtype=config.DTYPE, blocksize=0
    ) as stream:
        for chunk in chunks:
            data = np.asarray(chunk, dtype=np.float32).reshape(-1, 1)
            if data.size == 0:
                continue
            if first:
                first = False
                if on_first_chunk is not None:
                    on_first_chunk(data[:, 0])
            stream.write(data)


def normalize_audio(data: np.ndarray, peak: float = 0.95) -> np.ndarray:
    if data.size == 0:
        return data
    max_val = float(np.max(np.abs(data))) or 1.0
    return (data / max_val * peak).astype(np.float32)


def wait_for_speech_then_silence(
    capture: AudioCapture,
    voice_threshold: float = config.VOICE_THRESHOLD,
    silence_threshold: float = config.SILENCE_THRESHOLD,
    silence_duration: float = config.SILENCE_DURATION,
    max_total: float = config.MAX_RECORD_SECONDS,
    poll_interval: float = 0.05,
) -> np.ndarray:
    """Espera voz y luego silencio. Retorna el audio capturado.

    La captura debe estar ya activa y con `begin_recording()` llamado.
    Termina cuando se detecta voz seguida de `silence_duration` segundos
    de silencio, o cuando se alcanza `max_total`.
    """
    started_speaking = False
    quiet_time = 0.0
    start = time.time()
    while time.time() - start < max_total:
        time.sleep(poll_interval)
        wf = capture.get_waveform()
        recent = wf[-int(0.3 * capture.sample_rate):]
        rms = float(np.sqrt(np.mean(recent ** 2)))
        if not started_speaking:
            if rms >= voice_threshold:
                started_speaking = True
                quiet_time = 0.0
            continue
        if rms < silence_threshold:
            quiet_time += poll_interval
            if quiet_time >= silence_duration:
                return capture.end_recording()
        else:
            quiet_time = 0.0
    return capture.end_recording()