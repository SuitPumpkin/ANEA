"""Orquestador principal de ANEA.

Conecta: wake word → STT → LLM → TTS, gestionando estados y emociones
en la GUI desde el hilo principal (thread-safe via gui.submit).
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import config
from .audio import AudioCapture, play_audio_stream
from .gui import ANEAGui, resolve_device
from .llm import LLMEngine
from .stt import STTEngine, TranscriptionResult
from .tts import TTSEngine, persona_for_voice
from .wakeword import WakeWordDetector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("anea")


class AssistantState:
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    ERROR = "error"


class ANEA:
    def __init__(self) -> None:
        config.ensure_dirs()
        self.gui = ANEAGui()
        self._device = resolve_device(config.INPUT_DEVICE)
        self.capture = AudioCapture(device=self._device)
        self.capture.add_listener(self._on_audio_chunk)
        self.stt = STTEngine()
        self.llm = LLMEngine()
        default_voice = config.resolve_user_default_voice()
        self.tts = TTSEngine(voice=default_voice)
        if self.tts.available_voices:
            self.llm.system_prompt = persona_for_voice(self.tts.voice)
            self.gui.set_voice_options(
                self.tts.available_voices, self.tts.voice, self.tts.voice_labels
            )
        self.state = AssistantState.IDLE
        self._state_lock = threading.Lock()
        self._busy = threading.Lock()
        self._speaking_thread: threading.Thread | None = None
        self._init_steps_done: list[str] = []

        self.gui.on_manual_activate(self._on_manual_activate)
        self.gui.on_device_changed(self._on_device_changed)
        self.gui.on_voice_changed(self._on_voice_changed)

    def _set_state(self, state: str) -> None:
        with self._state_lock:
            self.state = state

    def _report_init_step(self, step: str, ok: bool, detail: str = "") -> None:
        status = "OK" if ok else f"FAIL ({detail})"
        log.info("Init step '%s': %s", step, status)
        if ok:
            self._init_steps_done.append(step)
        self.gui.submit(
            lambda s=step, st=status, d=list(self._init_steps_done): self.gui.set_loading(
                f"Inicializando {s}... {st}", d
            )
        )

    def _on_audio_chunk(self, _chunk) -> None:
        if not self.gui.is_shutting_down and self.state == AssistantState.LISTENING:
            self.capture.append_recording(_chunk)
            self.gui.push_waveform(_chunk)

    def _on_manual_activate(self) -> None:
        if self.state == AssistantState.LISTENING:
            self.gui.submit(self._finish_listening)
        else:
            self.gui.submit(self._activate_listening)

    def _on_wake_detected(self, _name: str) -> None:
        self.gui.submit(lambda: self._activate_listening())

    def _on_device_changed(self, device: int | str | None) -> None:
        log.info("Cambio de micrófono a: %s", device)
        self._device = device
        self.capture.set_device(device)
        self.gui._refresh_devices()

    def _on_voice_changed(self, voice: str) -> None:
        log.info("Cambio de voz a: %s", voice)
        prev = self.tts.voice
        self.tts.set_voice(voice)
        self.llm.system_prompt = persona_for_voice(voice)
        if prev != voice:
            self._ensure_voice_loaded(voice)

    def _ensure_voice_loaded(self, voice: str) -> None:
        try:
            self.tts.warmup(voice=voice)
        except Exception as exc:
            log.warning("Carga de voz '%s' falló: %s", voice, exc)

    def _activate_listening(self) -> None:
        if self.gui.is_shutting_down:
            return
        if not self._busy.acquire(blocking=False):
            log.info("Ignorando wake: asistente ocupado")
            return
        try:
            self._set_state(AssistantState.LISTENING)
            self.gui._set_emotion("listening")
            self.gui.show_status("Escuchando... pulsa TERMINAR cuando acabes")
            self.gui.set_recording(True)
            self.gui._clear_heard()
            self.capture.start()
            self.capture.begin_recording()
        except Exception:
            self._busy.release()
            raise

    def _finish_listening(self) -> None:
        if self.state != AssistantState.LISTENING:
            return
        try:
            audio = self.capture.end_recording()
            self.capture.stop()
            self.gui.set_recording(False)
            self.gui.show_status("Transcribiendo...")
            self.gui._set_emotion("thinking")
            self._set_state(AssistantState.TRANSCRIBING)
            threading.Thread(
                target=self._transcribe,
                args=(audio,),
                daemon=True,
                name="transcribe",
            ).start()
        except Exception as exc:
            log.exception("Error al terminar captura")
            self._fail(f"Error de captura: {exc}")

    def _handle_captured_audio(self, audio) -> None:
        self._set_state(AssistantState.TRANSCRIBING)
        self.gui._set_emotion("thinking")
        self.gui.show_status("Transcribiendo...")
        self.gui.set_recording(False)
        threading.Thread(
            target=self._transcribe,
            args=(audio,),
            daemon=True,
            name="transcribe",
        ).start()

    def _transcribe(self, audio) -> None:
        try:
            result: TranscriptionResult = self.stt.transcribe(audio)
        except Exception as exc:
            log.exception("Error STT")
            self._fail(f"Error de transcripción: {exc}")
            return
        if not result.text:
            self._fail("No entendí lo que dijiste.")
            return
        self.gui.submit(lambda t=result.text: self._dispatch_to_llm(t))

    def _dispatch_to_llm(self, text: str) -> None:
        """Muestra brevemente lo entendido y lanza la respuesta LLM+TTS."""
        self.gui.show_heard(text)
        self._set_state(AssistantState.THINKING)
        self.gui._set_emotion("thinking")
        self.gui.show_status("Pensando...")
        threading.Thread(
            target=self._think_and_speak,
            args=(text,),
            daemon=True,
            name="think-speak",
        ).start()

    def _think_and_speak(self, user_text: str) -> None:
        try:
            self._set_state(AssistantState.THINKING)
            self.gui.submit(lambda: (
                self.gui._set_emotion("thinking"),
                self.gui.show_status("Pensando..."),
            ))
            reply = self.llm.generate(user_text)
            log.info("LLM raw reply (%d chars): %r", len(reply), reply)
            reply = self._clean_reply(reply)
            log.info("LLM cleaned reply (%d chars): %r", len(reply), reply)
            if not reply.strip():
                # Si tras la limpieza quedó vacío (p.ej. el modelo solo
                # generó razonamiento interno sin responder), dar un
                # fallback breve y útil en lugar de fallar.
                log.warning("LLM respuesta vacía tras limpieza, usando fallback")
                reply = "Disculpa, no te entendí bien. ¿Puedes repetirlo?"
            self._set_state(AssistantState.SPEAKING)
            self.gui.submit(lambda r=reply: (
                self.gui.show_status("Hablando..."),
                self.gui._set_emotion("speaking"),
            ))
            self._synthesize_and_play(reply)
        except Exception as exc:
            log.exception("Error en think/speak")
            self._fail(f"Error: {exc}")
        else:
            self.gui.submit(self.gui.stop_speaking_animation)
            self._reset_after_turn()
        finally:
            try:
                self._busy.release()
            except Exception:
                pass

    @staticmethod
    def _clean_reply(text: str) -> str:
        """Limpia la respuesta del LLM para que TTS la lea fluidamente.

        - Quita bloques <think>...</think> (razonamiento interno del modelo).
        - Quita paréntesis y corchetes (suelen contener traducciones o aclaraciones).
        - Quita comillas y asteriscos de markdown.
        - Quita emojis y cualquier carácter unicode no-ASCII (defensa contra
          glitches o cortes de audio en el TTS).
                - Une las líneas para que una respuesta con formato de párrafo o lista
                    no se corte al pasarla al TTS.
        """
        import re

        if not text:
            return text
        # Quitar bloques de razonamiento interno <think>...</think>
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Quitar también bloques <think> sin cerrar al inicio (algunos
        # modelos dejan colgado el tag de cierre)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        # Quitar cualquier contenido entre paréntesis o corchetes
        text = re.sub(r"\([^)]*\)", "", text)
        text = re.sub(r"\[[^\]]*\]", "", text)
        # Quitar comillas tipográficas y asteriscos
        text = text.replace("“", "").replace("”", "")
        text = text.replace("‘", "").replace("’", "")
        text = text.replace("*", "").replace("#", "")
        # Quitar emojis y pictogramas
        text = re.sub(
            r"[\U0001F300-\U0001FAFF\U0001F600-\U0001F64F"
            r"\U0001F900-\U0001F9FF\u2600-\u26FF\u2700-\u27BF\uFE0F\u200D]",
            "",
            text,
        )
        # Quitar cualquier otro carácter unicode que no sea ASCII imprimible
        # o letras acentuadas españolas.
        text = re.sub(r"[^\x20-\x7EáéíóúÁÉÍÓÚñÑüÜ¿¡.,;:?!\n]", "", text)
        # Las respuestas pueden contener párrafos, chistes o listas. El TTS
        # debe recibir todo el contenido, no solo la primera línea.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return re.sub(r"\s+", " ", " ".join(lines)).strip()

    def _synthesize_and_play(self, text: str) -> None:
        log.info("TTS synthesizing (%d chars): %r", len(text), text)
        first_chunk = True

        def on_first_chunk(chunk: np.ndarray) -> None:
            nonlocal first_chunk
            if not first_chunk:
                return
            first_chunk = False
            log.info("TTS first chunk ready: %d samples", chunk.size)
            self.gui.submit(
                lambda c=chunk: self.gui.start_speaking_animation(
                    duration_s=c.size / config.TTS_SAMPLE_RATE, audio=c
                )
            )

        try:
            chunks = self.tts.synthesize_chunks(text)
            play_audio_stream(
                chunks,
                sample_rate=config.TTS_SAMPLE_RATE,
                on_first_chunk=on_first_chunk,
            )
        except Exception as exc:
            log.exception("Error TTS/reproducción")
            raise RuntimeError(f"TTS o reproducción falló: {exc}") from exc
        log.info("TTS stream complete")
        log.info("TTS play complete")

    def _reset_after_turn(self, status: str = 'Esperando "Hola Anea"...') -> None:
        self._set_state(AssistantState.IDLE)
        self.gui.submit(lambda s=status: (
            self.gui._set_emotion("neutral"),
            self.gui.show_status(s),
        ))

    def _fail(self, message: str) -> None:
        self._set_state(AssistantState.ERROR)
        self.gui.submit(lambda m=message: self.gui.show_error(m))
        try:
            self._busy.release()
        except Exception:
            pass

    def run(self) -> None:
        log.info("Inicializando ANEA...")
        threading.Thread(target=self._initialize_components, daemon=True, name="init").start()
        self.gui.mainloop()
        self.wakeword.stop()
        self.capture.stop()

    def _initialize_components(self) -> None:
        try:
            self._report_init_step("micrófono", True)
            self.capture.start()
            self._report_init_step("captura de audio", True)
            self._report_init_step("STT (Whisper)", True)
            try:
                self.stt.transcribe(np.zeros(int(0.1 * config.SAMPLE_RATE_INPUT), dtype=np.float32))
            except Exception as exc:
                log.warning("STT preload falló: %s", exc)
            model_label = f"LLM ({self.llm.model_path.name})"
            self._report_init_step(model_label, True)
            try:
                self.llm.generate("hola")
            except Exception as exc:
                log.warning("LLM preload falló: %s", exc)
            for voice in self.tts.available_voices:
                label = f"Voz '{self.tts.voice_labels.get(voice, voice)}' (Kokoro)"
                self._report_init_step(label, True)
                try:
                    self.tts.warmup(voice=voice)
                except Exception as exc:
                    log.warning("Warmup voz '%s' falló: %s", voice, exc)
                    self._report_init_step(label, False, str(exc))
            self.wakeword = WakeWordDetector(on_detect=self._on_wake_detected)
            self.wakeword.start()
            self._report_init_step("wake word", True)
            self.gui.submit(self.gui.set_ready)
            log.info("ANEA lista para usar")
        except Exception as exc:
            log.exception("Error durante inicialización: %s", exc)
            self.gui.submit(
                lambda: self.gui.show_error(f"Error al iniciar: {exc}", recover_ms=600000)
            )


def _parse_args() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(prog="anea", add_help=True)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Micrófono a usar (substring del nombre o índice). Por defecto: sistema.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Lista los dispositivos de entrada disponibles y sale.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Lista los modelos GGUF en Models/ y sale.",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="Lista las voces disponibles y sale.",
    )
    parser.add_argument(
        "--set-model",
        type=str,
        default=None,
        metavar="ARCHIVO",
        help="Guarda este modelo GGUF como el activo en anea_config.json.",
    )
    parser.add_argument(
        "--set-voice",
        type=str,
        default=None,
        metavar="VOZ",
        help="Guarda esta voz como la inicial en anea_config.json.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Carga los modelos pero no ejecuta TTS warmup (más rápido, primera respuesta lenta).",
    )
    args = parser.parse_args()

    if args.list_devices:
        from .gui import list_input_devices

        for i, name, ch in list_input_devices():
            print(f"  [{i}] {name} ({ch} in)")
        sys.exit(0)

    if args.list_models:
        from .config import discover_gguf_models
        for p in discover_gguf_models():
            print(f"  {p.name}  ({p.stat().st_size / 1024 / 1024:.0f} MB)")
        sys.exit(0)

    if args.list_voices:
        from .tts import TTSEngine
        e = TTSEngine()
        for v in e.available_voices:
            print(f"  {v}")
        sys.exit(0)

    if args.set_model or args.set_voice:
        cfg_path = config.USER_CONFIG_PATH
        existing = {}
        if cfg_path.exists():
            try:
                existing = json.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
        if args.set_model:
            existing["model"] = args.set_model
            print(f"Modelo por defecto: {args.set_model}")
        if args.set_voice:
            existing["default_voice"] = args.set_voice
            print(f"Voz por defecto: {args.set_voice}")
        cfg_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"Guardado en {cfg_path}")
        sys.exit(0)

    if args.device is not None:
        try:
            config.INPUT_DEVICE = int(args.device)
        except ValueError:
            config.INPUT_DEVICE = args.device
    if args.no_warmup:
        config._SKIP_WARMUP = True


def main() -> int:
    _parse_args()
    try:
        ANEA().run()
    except KeyboardInterrupt:
        log.info("Interrumpido por el usuario")
        return 0
    except Exception as exc:
        log.exception("Error fatal: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())