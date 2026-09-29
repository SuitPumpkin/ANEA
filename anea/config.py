"""Configuración central de ANEA.

Configuración técnica (rutas, parámetros de modelos) vive aquí.
Preferencias del usuario (modelo activo, voz por defecto) viven en
``anea_config.json`` en la raíz del proyecto y se cargan al importar.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Preferencias de usuario (cargadas desde JSON si existe)
USER_CONFIG_PATH = PROJECT_ROOT / "anea_config.json"

EMOTIONS_DIR = PROJECT_ROOT / "Emotions"
MODELS_DIR = PROJECT_ROOT / "Models"
VOICE_SAMPLE_DIR = PROJECT_ROOT / "Voice Samples"
TEMP_DIR = PROJECT_ROOT / "temp"

EMOTIONS = {
    "neutral": EMOTIONS_DIR / "Neutral.png",
    "listening": EMOTIONS_DIR / "Listening.png",
    "thinking": EMOTIONS_DIR / "Thinking.png",
    "speaking": EMOTIONS_DIR / "Speaking.png",
    "embarrassed": EMOTIONS_DIR / "Embarasement.png",
}

EMOTION_SIZE = 220

# Default: el primer modelo GGUF disponible por orden alfabético;
# el JSON del usuario puede sobreescribirlo.
GGUF_MODEL_PATH = MODELS_DIR / "qwen2.5-1.5b-instruct-q4_k_m.gguf"

LLM_SYSTEM_PROMPT = (
    "Responde EXCLUSIVAMENTE en español. Nunca uses inglés ni otro idioma. "
    "Nunca agregues traducciones entre paréntesis. Tu salida debe ser texto puro en español.\n"
    "Límite estricto: 1 sola frase corta (máximo 20 palabras). Sin listas, sin viñetas, sin markdown, "
    "sin emojis, sin comillas.\n"
    "Si no sabes algo, di 'No estoy segura' o 'No lo sé' en español.\n"
    "No inventes datos. Si te preguntan algo personal, responde brevemente sin repetir tu nombre.\n"
    "Tu nombre es Anea."
)

LLM_N_CTX = 2048
LLM_N_THREADS = max(1, (os.cpu_count() or 4) // 2)
LLM_N_GPU_LAYERS = -1
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 64
LLM_REPEAT_PENALTY = 1.15
LLM_TOP_P = 0.9
LLM_TOP_K = 40

# small reduce latencia manteniendo buena precisión en español; usa medium si
# la prioridad es máxima precisión y la GPU tiene margen suficiente.
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "auto"
WHISPER_COMPUTE_TYPE = "auto"
WHISPER_BEAM_SIZE = 1
WHISPER_BEST_OF = 1
WHISPER_VAD_FILTER = False
# Prompt inicial para Whisper: le da contexto sobre qué vocabulario esperar
# (nombres propios como "Anea", palabras en español, etc.) y mejora
#大幅 la precisión de transcripción.
WHISPER_INITIAL_PROMPT = (
    "Hola Anea, cuéntame un chiste, por favor. Háblame en español. "
    "Buenos días, ¿cómo estás? Necesito ayuda con algo, ¿puedes ayudarme?"
)
WHISPER_LANGUAGE = "es"

WAKEWORD_THRESHOLD = 0.5
WAKEWORD_MODEL_NAME = "hey_jarvis"
WAKEWORD_CHANNELS = 1
WAKEWORD_SAMPLE_RATE = 16000
WAKEWORD_ENABLED = False

TTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SPEAKER_EMBEDDING_PATH = TEMP_DIR / "speaker_embedding.pt"
SPEAKER_EMBEDDING_DIR = TEMP_DIR / "speaker_embeddings"
SILERO_MIKU_SPEAKER = "es_2"

# Voz por defecto (sobreescribible por anea_config.json).
DEFAULT_VOICE = "anea"
TTS_LANGUAGE = "es"
TTS_SAMPLE_RATE = 24000
TTS_CHUNK_MAX_CHARS = 120
TTS_CHUNK_PAUSE_SECONDS = 0.06

SAMPLE_RATE_INPUT = 16000
SAMPLE_RATE_OUTPUT = 24000
CHANNELS = 1
DTYPE = "float32"
INPUT_DEVICE = -1  # -1 = default; pass device index or name (str) to override

EDIT_TIMEOUT_SECONDS = 0.0
HEARD_DISPLAY_MS = 2500
WAKE_TO_SILENCE_SECONDS = 1.5
MAX_RECORD_SECONDS = 30.0
SPEAKING_ALTERNATE_MS = 500
ERROR_RECOVERY_MS = 2500
VOICE_THRESHOLD = 0.015
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 0.7

GUI_TITLE = "ANEA"
GUI_WIDTH = 520
GUI_HEIGHT = 720
GUI_THEME = "dark"
GUI_COLOR_THEME = "blue"


def ensure_dirs() -> None:
    """Crea carpetas necesarias (temp/, etc.) si no existen."""
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    SPEAKER_EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)


def discover_voice_samples() -> list[Path]:
    """Retorna todos los .wav en VOICE_SAMPLE_DIR ordenados alfabéticamente."""
    if not VOICE_SAMPLE_DIR.exists():
        return []
    return sorted(VOICE_SAMPLE_DIR.glob("*.wav"))


def discover_gguf_models() -> list[Path]:
    """Retorna todos los .gguf en MODELS_DIR ordenados alfabéticamente."""
    if not MODELS_DIR.exists():
        return []
    return sorted(MODELS_DIR.glob("*.gguf"))


def load_user_config() -> dict:
    """Carga preferencias del usuario desde anea_config.json si existe."""
    if not USER_CONFIG_PATH.exists():
        return {}
    try:
        with USER_CONFIG_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            log.warning("anea_config.json no es un objeto, ignorando")
            return {}
        return data
    except Exception as exc:
        log.warning("No se pudo leer anea_config.json: %s", exc)
        return {}


def resolve_user_model_path() -> Path:
    """Resuelve el path del modelo: usa anea_config.json si está, sino default."""
    user_cfg = load_user_config()
    model_name = user_cfg.get("model")
    if model_name:
        candidate = MODELS_DIR / model_name
        if candidate.exists():
            return candidate
        log.warning("Modelo '%s' de anea_config.json no existe, usando default", model_name)
    return GGUF_MODEL_PATH


def resolve_user_default_voice() -> str:
    """Resuelve la voz por defecto: usa anea_config.json si está, sino default."""
    user_cfg = load_user_config()
    voice = user_cfg.get("default_voice")
    if voice and isinstance(voice, str):
        return voice.lower()
    return DEFAULT_VOICE