# ANEA — Asistente de voz local

Asistente personal por voz para Windows con wake word "Hola Anea", STT local
(faster-whisper), LLM local (GGUF) y TTS con voces **sintéticas pre-entrenadas**
(Kokoro-82M). Todo se ejecuta en tu equipo: no hay servicios remotos.

**No hay clonación de voz.** Ver [`NOTICE.md`](NOTICE.md).

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
│   ├── fasttts.py      # Kokoro-82M (backend principal)
│   ├── tts.py          # TTSEngine: Kokoro + fallback Silero + personas
│   ├── gui.py          # CustomTkinter
│   └── main.py         # Orquestador
├── Emotions/           # PNGs: Neutral/Listening/Thinking/Speaking/Embarasement
├── Models/             # qwen*.gguf, mistral*.gguf, gemma*.gguf
├── Voice Samples/      # material local, ignorado por git. YA NO se usa en ejecución
├── temp/               # artefactos de prueba (ignorado por git)
├── anea_config.json    # Preferencias de usuario (modelo y voz por defecto)
├── NOTICE.md           # Datos de voz. Leer antes de publicar nada
├── requirements.txt
├── anea.bat            # Launcher (usa .venv311)
└── scripts/install.bat # Crea el venv Python 3.11 e instala todo
```

## Voces

Dos voces sintéticas, definidas en `TTS_VOICES` (`anea/config.py`):

| id | Label en la GUI | Voz Kokoro | Backend |
|---|---|---|---|
| `dora` | Dora | `ef_dora` | Kokoro-82M (GPU si está disponible) |
| `maria` | María | `em_santa` | Kokoro-82M (GPU si está disponible) |

Si Kokoro no se puede cargar, cada voz cae automáticamente a **Silero v3** y el
cambio queda registrado en el log. No hace falta instalar nada más.

`Voice Samples/` ya **no se lee** en tiempo de ejecución. Los `.wav` siguen en
disco y en `.gitignore` por si los necesitas, pero ANEA no los usa.

## Instalación

Todo va en un único venv de Python 3.11 (Kokoro no soporta 3.13):

```cmd
scripts\install.bat
```

Eso crea `.venv311` e instala GUI, audio, whisper, openWakeWord, llama-cpp-python,
Kokoro y Silero. El primer arranque descarga los pesos de Kokoro-82M (~330 MB,
se cachean en `~/.cache/huggingface`).

## Ejecución

```cmd
anea.bat
```

El wake word está deshabilitado por defecto (`WAKEWORD_ENABLED = False` en
`anea/config.py`); usa el botón "Activar manualmente" hasta tener un modelo
entrenado para "Hola Anea".

## Preferencias de usuario (`anea_config.json`)

```cmd
REM Ver modelos y voces disponibles
anea.bat --list-models
anea.bat --list-voices

REM Cambiar el modelo y la voz por defecto
anea.bat --set-model qwen2.5-1.5b-instruct-q4_k_m.gguf
anea.bat --set-voice maria
```

```json
{
  "model": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
  "default_voice": "dora"
}
```

- `model`: nombre exacto del `.gguf` en `Models/`.
- `default_voice`: id de voz en minúsculas (`dora` o `maria`).

> **Modelo recomendado: `qwen2.5-1.5b-instruct-q4_k_m.gguf`.** Es el único de los
> tres modelos incluidos que sigue la persona (responde en español, se identifica
> como Anea y no usa markdown). Gemma-3-1B se pasa al inglés y se identifica como
> Gemma; Mistral-7B responde en español pero añade traducciones al inglés y tarda
> bastante más.

Si el modelo o la voz del JSON no existen, ANEA usa el default de `config.py`.

## Configuración

Edita `anea/config.py`:

- Modelo Whisper (`WHISPER_MODEL_SIZE`): `tiny`/`base`/`small`/`medium`/`large-v3`
- Idioma STT (`WHISPER_LANGUAGE`)
- Wake word (`WAKEWORD_ENABLED`, `WAKEWORD_MODEL_NAME`, `WAKEWORD_THRESHOLD`)
- Voces (`TTS_VOICES`, `DEFAULT_VOICE_ID`)
- Kokoro (`KOKORO_USE_GPU`, `KOKORO_SPEED`)
- Longitud de respuesta (`LLM_MAX_TOKENS`, `TTS_CHUNK_MAX_CHARS`)

## Rendimiento

- Whisper usa `small`, GPU cuando está disponible, decodificación greedy y audio
  directo desde memoria. Para más precisión, `WHISPER_MODEL_SIZE = "medium"`.
- Kokoro-82M corre en GPU si `torch.cuda.is_available()`. Medido en una GTX 1660
  SUPER: 12 s de audio en 0.65 s de síntesis (rtf ≈ 0.05).
- La reproducción es incremental: ANEA mantiene abierto el dispositivo de salida
  y reproduce cada fragmento en cuanto está listo, sin esperar a sintetizar toda
  la respuesta.
- Las respuestas se pueden dividir en fragmentos de hasta
  `TTS_CHUNK_MAX_CHARS` (400) caracteres, con una pausa de 60 ms entre ellos.
- La primera respuesta tras arrancar es la más lenta: se cargan Whisper, el GGUF
  y Kokoro en memoria.

## Errores conocidos

- El wake word activo sigue siendo `hey_jarvis` y está deshabilitado. Falta
  entrenar el modelo de "Hola Anea" con openWakeWord.
- Los modelos de 1B (Qwen 1.5B, Gemma 3 1B) no son fiables obedecendo
  instrucciones estrictas. Se nota sobre todo en la Fase 4 (tareas en PC):
  un LLM de 1B puede inventar llamadas a herramientas o elegir mal una.
- Ejecutar tareas en la PC sigue deshabilitado. Falta el sistema de permisos
  (`Fase 4` del plan).

## Licencia y ética

- Código bajo licencia **MIT** (ver `LICENSE`).
- Los pesos de los modelos **no** se distribuyen aquí: tienen licencias propias.
  Kokoro-82M es Apache-2.0; los `.gguf` de Qwen/Mistral/Gemma tienen las suyas.
- **ANEA no clona voces.** Usa únicamente voces sintéticas pre-entrenadas. Lee
  [`NOTICE.md`](NOTICE.md) antes de publicar nada.
