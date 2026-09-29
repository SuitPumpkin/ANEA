"""GUI principal de ANEA con CustomTkinter."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter as tk
from collections import deque
from typing import Any, Callable

import customtkinter as ctk
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw, ImageTk

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
        self.root.resizable(False, False)
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.withdraw()
        self.root.configure(fg_color="#252525")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._emotion_images: dict[str, ImageTk.PhotoImage] = {}
        self._current_emotion = "neutral"
        self._waveform_data: deque[float] = deque(maxlen=config.SAMPLE_RATE_INPUT)
        self._speaking_timer: str | None = None
        self._speaking_levels = np.zeros(1, dtype=np.float32)
        self._speaking_frame = 0
        self._error_after_id: str | None = None
        self._waveform_after_id: str | None = None
        self._panel_animation_id: str | None = None
        self._tray_icon: Any = None
        self._panel_visible = False
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
        self._start_tray_icon()
        self.root.after(120, self.show_panel)

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown.is_set()

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _build_layout(self) -> None:
        self._main_frame = ctk.CTkFrame(
            self.root, corner_radius=12, fg_color="#252525", border_width=1,
            border_color="#3d3d3d"
        )
        self._main_frame.pack(fill="both", expand=True, padx=26, pady=18)

        top_bar = ctk.CTkFrame(self._main_frame, fg_color="transparent", height=44)
        top_bar.pack(fill="x", padx=24, pady=(14, 0))
        top_bar.pack_propagate(False)
        brand = ctk.CTkFrame(top_bar, fg_color="transparent")
        brand.pack(side="left", anchor="center")
        ctk.CTkLabel(
            brand, text="ANEA", text_color="#f5f2ed", font=("Segoe UI", 18, "bold")
        ).pack(anchor="w")
        ctk.CTkLabel(
            brand, text="ASISTENTE DE VOZ", text_color="#777977",
            font=("Segoe UI", 8, "bold")
        ).pack(anchor="w", pady=(1, 0))
        self.close_btn = ctk.CTkButton(
            top_bar, text="×", width=32, height=32, corner_radius=8,
            fg_color="transparent", hover_color="#493033", text_color="#c4555b",
            font=("Segoe UI", 27), command=self._on_close,
        )
        self.close_btn.pack(side="right")

        avatar_stage = ctk.CTkFrame(
            self._main_frame, width=292, height=292, corner_radius=0,
            fg_color="transparent", border_width=0,
        )
        avatar_stage.pack(pady=(6, 8))
        avatar_stage.pack_propagate(False)
        self.soundwave_canvas = tk.Canvas(
            avatar_stage, width=292, height=292, bg="#252525", highlightthickness=0
        )
        self.soundwave_canvas.place(x=0, y=0)
        self.avatar_canvas = tk.Canvas(
            avatar_stage, width=244, height=244, bg="#252525", highlightthickness=0
        )
        self.avatar_canvas.place(relx=0.5, rely=0.5, anchor="center")
        self.soundwave_canvas.place_forget()

        self.status_label = ctk.CTkLabel(
            self._main_frame, text="Inicializando...", font=("Segoe UI", 11, "bold"),
            text_color="#b7bab6", height=24,
        )
        self.status_label.pack(pady=(0, 6))

        self.waveform = ctk.CTkCanvas(
            self._main_frame, height=48, bg="#252525", highlightthickness=0
        )
        self.waveform.pack(fill="x", padx=28, pady=(4, 12))

        self.heard_label = ctk.CTkLabel(
            self._main_frame, text="Habla con ANEA...", font=("Consolas", 12),
            text_color="#f5f2ed", fg_color="#464646", corner_radius=10,
            wraplength=config.GUI_WIDTH - 120, justify="center", height=54,
        )
        self.heard_label.pack(pady=(0, 12), padx=28, fill="x")
        self._heard_clear_after: str | None = None

        selectors_row = ctk.CTkFrame(self._main_frame, fg_color="transparent")
        selectors_row.pack(fill="x", padx=28, pady=(0, 8))
        device_column = ctk.CTkFrame(selectors_row, fg_color="transparent")
        device_column.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkLabel(
            device_column, text="MICRÓFONO", text_color="#777977",
            font=("Segoe UI", 8, "bold"), anchor="w"
        ).pack(fill="x", pady=(0, 3))

        voice_column = ctk.CTkFrame(selectors_row, fg_color="transparent", width=138)
        voice_column.pack(side="right", fill="y")
        ctk.CTkLabel(
            voice_column, text="VOZ", text_color="#777977",
            font=("Segoe UI", 8, "bold"), anchor="w"
        ).pack(fill="x", pady=(0, 3))

        self.manual_btn = ctk.CTkButton(
            self._main_frame, text="▶  INICIAR", width=132, height=46, corner_radius=8,
            fg_color="#5b5b5b", hover_color="#707070", font=("Segoe UI", 19),
            command=self._on_manual_click, state="disabled",
        )

        self.device_var = tk.StringVar(value="(default)")
        self.device_menu = ctk.CTkOptionMenu(
            device_column, variable=self.device_var, values=["(default)"],
            command=self._on_device_selected, width=220, height=32,
            corner_radius=8, fg_color="#454545", button_color="#454545",
            button_hover_color="#5b5b5b", text_color="#ece9e4",
        )
        self.device_menu.pack(fill="x")

        self.voice_var = tk.StringVar(value="anea")
        self.voice_menu = ctk.CTkOptionMenu(
            voice_column, variable=self.voice_var, values=["anea", "miku"],
            command=self._on_voice_selected, width=138, height=32,
            corner_radius=8, fg_color="#454545", button_color="#454545",
            button_hover_color="#5b5b5b", text_color="#ece9e4",
        )
        self.voice_menu.pack(fill="x")
        self.manual_btn.pack(anchor="e", padx=28, pady=(0, 10))

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
        size = (238, 238)
        for key in self.EMOTION_KEYS:
            path = config.EMOTIONS[key]
            if not path.exists():
                log.warning("Falta emoción %s: %s", key, path)
                continue
            img = Image.open(path).convert("RGBA").resize(size, Image.Resampling.LANCZOS)
            mask = Image.new("L", size, 0)
            ImageDraw.Draw(mask).ellipse((0, 0, size[0] - 1, size[1] - 1), fill=255)
            circular = Image.new("RGBA", size, "#252525")
            circular.paste(img, (0, 0), mask)
            self._emotion_images[key] = ImageTk.PhotoImage(circular)

    def _set_emotion(self, key: str) -> None:
        if key not in self._emotion_images:
            return
        self._current_emotion = key
        self.avatar_canvas.delete("all")
        self.avatar_canvas.create_oval(
            1, 1, 243, 243, fill="#303d75", outline="#303d75"
        )
        self.avatar_canvas.create_image(122, 122, image=self._emotion_images[key])
        if key != "speaking":
            self.soundwave_canvas.place_forget()

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

    def set_recording(self, active: bool) -> None:
        """Actualiza el control principal para iniciar o terminar una toma."""
        if active:
            self.manual_btn.configure(
                text="■  TERMINAR", fg_color="#b7474c", hover_color="#d15b61"
            )
        else:
            self.manual_btn.configure(
                text="▶  INICIAR", fg_color="#5b5b5b", hover_color="#707070"
            )

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

    def _start_tray_icon(self) -> None:
        """Crea el icono de bandeja sin bloquear el hilo de Tk."""
        try:
            import pystray
        except ImportError:
            log.warning("pystray no está instalado; el panel seguirá disponible sin bandeja")
            return

        icon_image = Image.new("RGB", (64, 64), (37, 37, 37))
        pixels = icon_image.load()
        for x in range(64):
            for y in range(64):
                distance = ((x - 32) ** 2 + (y - 32) ** 2) ** 0.5
                if 20 <= distance <= 27:
                    pixels[x, y] = (145, 198, 75)
                elif distance < 20:
                    pixels[x, y] = (48, 61, 117)

        menu = pystray.Menu(
            pystray.MenuItem("Mostrar ANEA", lambda: self.root.after(0, self.show_panel)),
            pystray.MenuItem("Salir", lambda: self.root.after(0, self.quit)),
        )
        self._tray_icon = pystray.Icon("anea", icon_image, "ANEA", menu)
        threading.Thread(
            target=self._tray_icon.run, daemon=True, name="anea-tray"
        ).start()

    def _position_panel(self, y: int) -> None:
        width = self.root.winfo_width() or config.GUI_WIDTH
        height = self.root.winfo_height() or config.GUI_HEIGHT
        x = self.root.winfo_screenwidth() - width - 24
        taskbar_gap = 48
        target_y = self.root.winfo_screenheight() - height - taskbar_gap
        self.root.geometry(f"{width}x{height}+{x}+{y if y >= 0 else target_y}")

    def show_panel(self) -> None:
        """Muestra el popover con una animación ascendente desde la bandeja."""
        if self._shutdown.is_set():
            return
        if self._panel_animation_id is not None:
            self.root.after_cancel(self._panel_animation_id)
        self.root.update_idletasks()
        width = self.root.winfo_width() or config.GUI_WIDTH
        height = self.root.winfo_height() or config.GUI_HEIGHT
        x = self.root.winfo_screenwidth() - width - 24
        target_y = self.root.winfo_screenheight() - height - 48
        start_y = self.root.winfo_screenheight() + 8
        self.root.geometry(f"{width}x{height}+{x}+{start_y}")
        self.root.deiconify()
        self.root.lift()
        self._panel_visible = True

        def animate(current_y: int) -> None:
            if self._shutdown.is_set():
                return
            next_y = max(target_y, current_y - 42)
            self.root.geometry(f"{width}x{height}+{x}+{next_y}")
            if next_y > target_y:
                self._panel_animation_id = self.root.after(12, lambda: animate(next_y))
            else:
                self._panel_animation_id = None

        animate(start_y)

    def _hide_panel(self) -> None:
        if self._panel_animation_id is not None:
            self.root.after_cancel(self._panel_animation_id)
            self._panel_animation_id = None
        self.root.withdraw()
        self._panel_visible = False

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
        self._hide_panel()

    def quit(self) -> None:
        """Cierra el proceso desde el menú de la bandeja."""
        self._shutdown.set()
        for after_id in (
            self._heard_clear_after, self._speaking_timer, self._error_after_id,
            self._waveform_after_id, self._panel_animation_id,
        ):
            if after_id is not None:
                try:
                    self.root.after_cancel(after_id)
                except Exception:
                    pass
        if self._tray_icon is not None:
            try:
                self._tray_icon.stop()
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

    def start_speaking_animation(
        self, duration_s: float, audio: np.ndarray | None = None
    ) -> None:
        if self._speaking_timer is not None:
            self.root.after_cancel(self._speaking_timer)
            self._speaking_timer = None
        if audio is None or audio.size == 0:
            self._speaking_levels = np.ones(1, dtype=np.float32) * 0.35
        else:
            samples_per_frame = max(1, int(audio.size / max(1, duration_s * 30)))
            levels = []
            for start in range(0, audio.size, samples_per_frame):
                chunk = audio[start:start + samples_per_frame]
                levels.append(float(np.sqrt(np.mean(np.square(chunk)))))
            self._speaking_levels = np.asarray(levels, dtype=np.float32)
            peak = float(np.max(self._speaking_levels)) or 1.0
            self._speaking_levels = np.clip(self._speaking_levels / peak, 0.12, 1.0)
        self._speaking_frame = 0
        self.soundwave_canvas.place(x=0, y=0)

        def tick() -> None:
            if self._shutdown.is_set():
                return
            level = float(self._speaking_levels[self._speaking_frame % len(self._speaking_levels)])
            self._draw_soundwave(level)
            self._speaking_frame += 1
            self._speaking_timer = self.root.after(33, tick)

        tick()

    def _draw_soundwave(self, level: float) -> None:
        self.soundwave_canvas.delete("all")
        center = 146
        radius = 128 + level * 3
        width = 5 + int(level * 9)
        self.soundwave_canvas.create_oval(
            center - radius, center - radius, center + radius, center + radius,
            outline="#91c64b", width=width,
        )
        self.soundwave_canvas.create_oval(
            center - radius - 5, center - radius - 5,
            center + radius + 5, center + radius + 5,
            outline="#567c35", width=2,
        )

    def stop_speaking_animation(self) -> None:
        if self._speaking_timer is not None:
            self.root.after_cancel(self._speaking_timer)
            self._speaking_timer = None
        self.soundwave_canvas.delete("all")
        self.soundwave_canvas.place_forget()
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