"""GUI principal de ANEA con CustomTkinter."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from collections import deque
from typing import Callable

import customtkinter as ctk
import numpy as np
import sounddevice as sd
from PIL import Image

from . import config

log = logging.getLogger(__name__)


def list_input_devices() -> list[tuple[int, str, int]]:
    """Retorna lista de (index, name, max_input_channels) para dispositivos con entrada."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log.warning("No se pudo enumerar dispositivos: %s", exc)
        return []
    result = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            result.append((i, d["name"], int(d["max_input_channels"])))
    return result


def resolve_device(device) -> int | str | None:
    """Resuelve el INPUT_DEVICE de config a un valor válido para sounddevice."""
    if device is None or device == -1:
        return None
    if isinstance(device, str):
        try:
            devices = sd.query_devices()
        except Exception:
            return None
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and device.lower() in d["name"].lower():
                return i
        log.warning("Micrófono '%s' no encontrado, usando default", device)
        return None
    return int(device)


class ANEAGui:
    """Ventana principal: estado de carga + emoción + waveform + transcripción."""

    EMOTION_KEYS = ("neutral", "listening", "thinking", "speaking", "embarrassed")

    def __init__(self) -> None:
        ctk.set_appearance_mode(config.GUI_THEME)
        ctk.set_default_color_theme("dark-blue")
        self.root = ctk.CTk()
        self.root.title(config.GUI_TITLE)
        self.root.geometry("520x720")
        self.root.minsize(440, 620)
        self.root.configure(fg_color="#f4f4f2")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._emotion_images: dict[str, ctk.CTkImage] = {}
        self._current_emotion = "neutral"
        self._waveform_data: deque[float] = deque(maxlen=config.SAMPLE_RATE_INPUT)
        self._speaking_timer: str | None = None
        self._error_after_id: str | None = None
        self._waveform_after_id: str | None = None
        self._command_queue: "queue.Queue[Callable[[], None]]" = queue.Queue()
        self._shutdown = threading.Event()
        self._ready = False
        self._on_transcription_ready: Callable[[str], None] | None = None
        self._on_manual_activate: Callable[[], None] | None = None
        self._on_device_changed: Callable[[int | str | None], None] | None = None
        self._on_voice_changed: Callable[[str], None] | None = None

        self._build_layout()
        self._build_loading_overlay()
        self._preload_emotions()
        self._set_emotion("neutral")
        self._poll_queue()
        self._refresh_devices()

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown.is_set()

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _build_layout(self) -> None:
        self._main_frame = ctk.CTkFrame(
            self.root, corner_radius=28, fg_color="#292929", border_width=0
        )
        self._main_frame.pack(fill="both", expand=True, padx=24, pady=16)

        top_bar = ctk.CTkFrame(self._main_frame, fg_color="transparent", height=38)
        top_bar.pack(fill="x", padx=22, pady=(12, 0))
        top_bar.pack_propagate(False)
        ctk.CTkLabel(
            top_bar, text="ANEA", text_color="#f6f1ea", font=("Segoe UI", 16, "bold")
        ).pack(side="left")
        self.close_btn = ctk.CTkButton(
            top_bar, text="×", width=32, height=32, corner_radius=16,
            fg_color="transparent", hover_color="#493033", text_color="#b7474c",
            font=("Segoe UI", 27), command=self._on_close,
        )
        self.close_btn.pack(side="right")

        self.avatar_frame = ctk.CTkFrame(
            self._main_frame, width=274, height=274, corner_radius=137,
            fg_color="#86b943", border_width=0,
        )
        self.avatar_frame.pack(pady=(16, 12))
        self.avatar_frame.pack_propagate(False)
        self.face_label = ctk.CTkLabel(
            self.avatar_frame, text="", fg_color="#3a4a86", corner_radius=124
        )
        self.face_label.pack(fill="both", expand=True, padx=8, pady=8)

        self.status_label = ctk.CTkLabel(
            self._main_frame, text="Inicializando...", font=("Segoe UI", 12),
            text_color="#a9aaa8",
        )
        self.status_label.pack(pady=(0, 4))

        self.waveform = ctk.CTkCanvas(
            self._main_frame, height=54, bg="#292929", highlightthickness=0
        )
        self.waveform.pack(fill="x", padx=32, pady=(8, 10))

        self.heard_label = ctk.CTkLabel(
            self._main_frame, text="", font=("Consolas", 12),
            text_color="#f5f2ed", fg_color="#505050", corner_radius=14,
            wraplength=config.GUI_WIDTH - 110, justify="center", height=54,
        )
        self.heard_label.pack(pady=(0, 10), padx=34, fill="x")
        self._heard_clear_after: str | None = None

        button_row = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        button_row.pack(fill="x", padx=34, pady=(0, 10))
        self.manual_btn = ctk.CTkButton(
            button_row, text="▶", width=48, height=48, corner_radius=14,
            fg_color="#555555", hover_color="#686868", font=("Segoe UI", 19),
            command=self._on_manual_click, state="disabled",
        )
        self.manual_btn.pack(side="right", padx=(10, 0))

        self.device_var = tk.StringVar(value="(default)")
        self.device_menu = ctk.CTkOptionMenu(
            button_row, variable=self.device_var, values=["(default)"],
            command=self._on_device_selected, width=150, height=32,
            corner_radius=16, fg_color="#505050", button_color="#505050",
            button_hover_color="#646464",
        )
        self.device_menu.pack(side="left", padx=(0, 8))

        self.voice_var = tk.StringVar(value="anea")
        self.voice_menu = ctk.CTkOptionMenu(
            button_row, variable=self.voice_var, values=["anea", "miku"],
            command=self._on_voice_selected, width=112, height=32,
            corner_radius=16, fg_color="#505050", button_color="#505050",
            button_hover_color="#646464",
        )
        self.voice_menu.pack(side="left")

        self._waveform_after_id = self.root.after(50, self._tick_waveform)

    def _build_loading_overlay(self) -> None:
        self._loading_frame = ctk.CTkFrame(self.root, corner_radius=0)
        self._loading_frame.place(relx=0, rely=0, relwidth=1, relheight=1)
        inner = ctk.CTkFrame(self._loading_frame, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        self._loading_title = ctk.CTkLabel(
            inner, text="ANEA", font=("", 36, "bold"),
        )
        self._loading_title.pack(pady=(0, 20))
        self._loading_label = ctk.CTkLabel(
            inner, text="Inicializando...", font=("", 16),
        )
        self._loading_label.pack(pady=(0, 12))
        self._progress = ctk.CTkProgressBar(inner, mode="indeterminate", width=320)
        self._progress.pack(pady=(0, 12))
        self._progress.start()
        self._loading_steps = ctk.CTkLabel(
            inner, text="", font=("", 12), justify="left", text_color="#aaaaaa",
        )
        self._loading_steps.pack()

    def _preload_emotions(self) -> None:
        size = (config.EMOTION_SIZE, config.EMOTION_SIZE)
        for key in self.EMOTION_KEYS:
            path = config.EMOTIONS[key]
            if not path.exists():
                log.warning("Falta emoción %s: %s", key, path)
                continue
            img = Image.open(path).convert("RGBA")
            self._emotion_images[key] = ctk.CTkImage(light_image=img, dark_image=img, size=size)

    def _set_emotion(self, key: str) -> None:
        if key not in self._emotion_images:
            return
        self._current_emotion = key
        self.face_label.configure(image=self._emotion_images[key])
        self.avatar_frame.configure(
            fg_color="#86b943" if key == "speaking" else "#292929"
        )

    def push_waveform(self, data: np.ndarray) -> None:
        if data.size == 0:
            return
        self._waveform_data.extend(data.astype(np.float32).tolist())

    def _tick_waveform(self) -> None:
        if self._shutdown.is_set():
            return
        self.waveform.delete("all")
        w = self.waveform.winfo_width()
        h = self.waveform.winfo_height()
        if w <= 2 or h <= 2:
            self._waveform_after_id = self.root.after(50, self._tick_waveform)
            return
        data = np.fromiter(self._waveform_data, dtype=np.float32, count=len(self._waveform_data))
        if data.size < 2:
            self._waveform_after_id = self.root.after(50, self._tick_waveform)
            return
        step = max(1, data.size // w)
        sampled = data[::step][:w]
        if sampled.size < w:
            sampled = np.concatenate([np.zeros(w - sampled.size, dtype=np.float32), sampled])
        mid = h / 2
        scale = (h / 2) * 0.9
        points: list[float] = []
        for x, v in enumerate(sampled):
            points.extend([x, mid - v * scale])
        if len(points) >= 4:
            self.waveform.create_line(points, fill="#23b8bd", width=2, smooth=True)
            self.waveform.create_line(
                [0, mid, w, mid], fill="#167276", width=1
            )
        self._waveform_after_id = self.root.after(50, self._tick_waveform)

    def show_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def show_heard(self, text: str) -> None:
        """Muestra brevemente lo que se entendió del usuario."""
        if not text:
            return
        self.heard_label.configure(text=f"«{text}»")
        if self._heard_clear_after is not None:
            self.root.after_cancel(self._heard_clear_after)
        self._heard_clear_after = self.root.after(
            config.HEARD_DISPLAY_MS, self._clear_heard
        )

    def _clear_heard(self) -> None:
        self._heard_clear_after = None
        self.heard_label.configure(text="")

    def _on_manual_click(self) -> None:
        if self._on_manual_activate is not None:
            self._on_manual_activate()

    def _refresh_devices(self) -> None:
        devices = list_input_devices()
        labels = ["(default)"] + [f"[{i}] {name}" for i, name, _ in devices]
        try:
            current = self.device_var.get()
        except Exception:
            current = "(default)"
        self.device_menu.configure(values=labels)
        if current not in labels:
            self.device_var.set("(default)")

    def _on_device_selected(self, choice: str) -> None:
        if self._on_device_changed is not None:
            if choice == "(default)":
                self._on_device_changed(None)
                return
            try:
                idx = int(choice.split("]")[0].lstrip("["))
                self._on_device_changed(idx)
            except (ValueError, IndexError):
                log.warning("Selección de micrófono inválida: %s", choice)

    def _on_close(self) -> None:
        self._shutdown.set()
        for after_id in (self._heard_clear_after, self._speaking_timer, self._error_after_id, self._waveform_after_id):
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
        self.root.destroy()

    def _poll_queue(self) -> None:
        if self._shutdown.is_set():
            return
        try:
            while True:
                fn = self._command_queue.get_nowait()
                try:
                    fn()
                except Exception as exc:
                    log.exception("Error ejecutando comando GUI: %s", exc)
        except queue.Empty:
            pass
        self.root.after(50, self._poll_queue)

    def submit(self, fn: Callable[[], None]) -> None:
        """Encola una acción para ejecutarse en el hilo principal."""
        self._command_queue.put(fn)

    def start_speaking_animation(self, duration_s: float) -> None:
        if self._speaking_timer is not None:
            self.root.after_cancel(self._speaking_timer)
            self._speaking_timer = None
        toggle = {"on": False}

        def tick() -> None:
            if self._shutdown.is_set():
                return
            toggle["on"] = not toggle["on"]
            self._set_emotion("speaking" if toggle["on"] else "neutral")
            self._speaking_timer = self.root.after(config.SPEAKING_ALTERNATE_MS, tick)

        tick()

    def stop_speaking_animation(self) -> None:
        if self._speaking_timer is not None:
            self.root.after_cancel(self._speaking_timer)
            self._speaking_timer = None
        self._set_emotion("neutral")

    def show_error(self, message: str, recover_ms: int = config.ERROR_RECOVERY_MS) -> None:
        if self._error_after_id is not None:
            self.root.after_cancel(self._error_after_id)
            self._error_after_id = None
        self._set_emotion("embarrassed")
        self.show_status(message)

        def recover() -> None:
            self._error_after_id = None
            self._set_emotion("neutral")
            self.show_status('Esperando "Oye Anea"...')

        self._error_after_id = self.root.after(recover_ms, recover)

    def on_transcription_ready(self, cb: Callable[[str], None]) -> None:
        self._on_transcription_ready = cb

    def on_manual_activate(self, cb: Callable[[], None]) -> None:
        self._on_manual_activate = cb

    def on_device_changed(self, cb: Callable[[int | str | None], None]) -> None:
        self._on_device_changed = cb

    def on_voice_changed(self, cb: Callable[[str], None]) -> None:
        self._on_voice_changed = cb

    def set_voice(self, name: str) -> None:
        if name in self.voice_menu.cget("values"):
            self.voice_var.set(name)

    def set_voice_options(self, voices: list[str], current: str) -> None:
        """Reemplaza la lista de voces disponibles (cloned + silero)."""
        self.voice_menu.configure(values=voices)
        if current in voices:
            self.voice_var.set(current)
        elif voices:
            self.voice_var.set(voices[0])

    def _on_voice_selected(self, choice: str) -> None:
        if self._on_voice_changed is not None:
            self._on_voice_changed(choice)

    def set_loading(self, message: str, done_steps: list[str] | None = None) -> None:
        self._loading_label.configure(text=message)
        if done_steps:
            self._loading_steps.configure(text="\n".join(f"✓ {s}" for s in done_steps))

    def set_ready(self) -> None:
        """Cierra el overlay y habilita la UI principal."""
        self._progress.stop()
        self._loading_frame.destroy()
        self.manual_btn.configure(state="normal")
        self._ready = True
        self.show_status('Esperando "Oye Anea"...')

    def mainloop(self) -> None:
        self.root.mainloop()