# ANEA — Mini Asistente Personal con VRAM

Asistente personal por voz para Windows con wake word "Oye Anea", STT local
(Spanish), LLM local (Qwen 1.5B Q4_K_M) y TTS con voz clonada (Coqui XTTS v2).

## Arquitectura de Estados

```
IDLE (Neutral) → LISTENING → TRANSCRIBING → THINKING → SPEAKING → IDLE
                                    ↘ EMBARRASSED (error) ↗
```

## Estructura

```
ANEA/
├── anea/
│   ├── __init__.py
│   ├── config.py       # Rutas y parámetros técnicos
│   ├── audio.py        # Captura mic + waveform + reproducción
│   ├── wakeword.py     # openWakeWord
│   ├── stt.py          # faster-whisper
│   ├── llm.py          # llama-cpp-python (GGUF)
│   ├── tts.py          # Coqui XTTS v2 + Silero + speaker embedding cache
│   ├── gui.py          # CustomTkinter
│   └── main.py         # Orquestador
├── Emotions/           # PNGs: Neutral/Listening/Thinking/Speaking/Embarasement
├── Models/             # qwen*.gguf, mistral*.gguf, etc.
├── Voice Samples/      # un .wav por voz clonada. Ver NOTICE.md: NO se versionan
├── temp/               # speaker_embeddings/{voz}.pt (auto-creado, no se versiona)
├── anea_config.json    # Preferencias de usuario (modelo y voz por defecto)
├── NOTICE.md           # Consentimiento para clonar voces. Leer antes de clonar ninguna
├── requirements.txt
├── anea.bat            # Launcher (usa .venv311)
└── scripts/install.bat # Crea venv Python 3.11 + instala TTS
```

> **Este repositorio no incluye ninguna muestra de voz.** Las muestras son datos biometricos y
> subirlas sin consentimiento escrito de la persona es ilegal en varias jurisdicciones. ANEA
> funciona sin ellas usando `miku` (la voz rapida de Silero, sin clonacion). Ver
> [`NOTICE.md`](NOTICE.md) y [`Voice Samples/README.md`](Voice%20Samples/README.md).

## Instalación

### 1) Python 3.13 — dependencias principales

```powershell
pip install -r requirements.txt
```

### 2) Python 3.11 — TTS (Coqui requiere Python <3.12)

```powershell
scripts\install.bat
```

Esto descarga e instala Python 3.11.9 en `C:\Python311` (si no existe) y
crea un venv `.venv311` con `TTS`.

### 3) Recursos

- `Emotions/`: PNGs ya presentes (nota: `Embarasement` con una sola 'r').
- `Model/qwen2.5-1.5b-instruct-q4_k_m.gguf`: ya presente.
- `Voice Sample/*.wav`: ya presente; el nombre se detecta automáticamente.

## Ejecución

```cmd
anea.bat
```

El wake word está deshabilitado por defecto (usa el botón "Activar manualmente").

## Preferencias de usuario (`anea_config.json`)

Para elegir qué modelo y qué voz se usan por defecto, edita
`anea_config.json` o usa los flags de línea de comandos:

```cmd
REM Ver los modelos y voces disponibles
anea.bat --list-models
anea.bat --list-voices

REM Cambiar el modelo y la voz por defecto (los guarda en anea_config.json)
anea.bat --set-model mistral-7b-instruct-v0.2.Q4_K_M.gguf
anea.bat --set-voice alex
```

Ejemplo de `anea_config.json`:

```json
{
  "model": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
  "default_voice": "anea"
}
```

- `model`: nombre exacto del archivo `.gguf` en `Models/`.
- `default_voice`: nombre de la voz (minúsculas) — puede ser el nombre de
  un `.wav` en `Voice Samples/` (sin extensión) o `miku` para la voz rápida
  de Silero. Para usar una voz Miku real, añade una muestra de voz **con
  consentimiento escrito de esa persona** como `Voice Samples/Miku.wav`; ANEA
  la cargará con XTTS. Sin esa muestra, `miku` usa el fallback genérico `es_2`
  de Silero.

> `default_voice: "miku"` funciona **sin ninguna muestra de voz** y sin clonar a
> nadie. Es la opción por defecto recomendada para probar el proyecto.

Si el modelo o la voz del JSON no existen, ANEA usa el default de
`config.py`.

## Configuración

Edita `anea/config.py` para ajustar:

- Modelo Whisper (`WHISPER_MODEL_SIZE`): `tiny`/`base`/`small`/`medium`/`large-v3`
- Idioma STT (`WHISPER_LANGUAGE`)
- Modelo GGUF (`GGUF_MODEL_PATH`), GPU layers (`LLM_N_GPU_LAYERS`)
- Idioma TTS (`TTS_LANGUAGE`)
- Tiempo de edición de transcripción (`EDIT_TIMEOUT_SECONDS`)

## Notas

- El primer arranque descargará el modelo de faster-whisper (~460MB para `small`).
- XTTS v2 descarga ~1.8GB la primera vez.
- Speaker embedding se cachea en `temp/speaker_embedding.pt`.
- Si XTTS carga sin GPU, la síntesis será muy lenta. Ajusta `gpu` en
  `TTSEngine.initialize()` según tu hardware.

## Licencia y ética

- Código bajo licencia **MIT** (ver `LICENSE`).
- Los pesos de los modelos (`.gguf`) **no** se distribuyen aquí: tienen licencias propias.
- **La clonación de voz requiere consentimiento escrito de la persona cuya voz se clona.**
  La voz es un dato biométrico. Lee [`NOTICE.md`](NOTICE.md) antes de usar esta capacidad.