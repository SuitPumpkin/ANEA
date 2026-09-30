# NOTICE

## Sin clonacion de voz

ANEA **no clona voces**. El TTS usa unicamente voces sinteticas pre-entrenadas de
[Kokoro-82M](https://github.com/hexgrad/kokoro) (82M parametros, pesos bajo
licencia Apache-2.0), con un respaldo en Silero v3 (MIT). Ninguna de las dos
identifica a una persona real: son modelos generativos entrenados con audio
sintetico, no con grabaciones de personas concretas.

Esto cambio respecto a la version anterior del proyecto, que usaba
**Coqui XTTS v2** y si clonaba la voz de quien estuviera grabada en
`Voice Samples/`. XTTS fue retirado de la base de codigo. Los `.wav` que quedan en
`Voice Samples/` ya no se leen en tiempo de ejecucion.

## Por que la regla sigue en pie

La voz de una persona es un **dato biometrico**. No es un archivo de audio
cualquiera: identifica a quien habla y, en muchas jurisdicciones, esta protegida
por leyes de proteccion de datos personales (por ejemplo, la LFPDPPP en Mexico y
el RGPD en la Union Europea, que exige consentimiento explicito para tratar datos
biometricos).

Aunque ANEA ya no clona, `Voice Samples/` sigue en `.gitignore` a proposito, y
sigue siendo una buena regla no subir audio de personas a ningun repositorio.

### Reglas del proyecto

1. **No subas tu `Voice Samples/` a este repositorio, ni a ningun otro.** Las
   muestras estan en `.gitignore` y deben seguir ahi.
2. **Usa la voz de una persona requiere consentimiento escrito de esa persona**,
   no solo permiso informal. "Me dejo usar su voz" no es una base legal
   suficiente. El consentimiento tiene que ser consciente, escrito, revocable y
   con fines delimitados.
3. **No uses voces de terceros en espacios publicos** como demostracion del
   proyecto.
4. **No abras excepciones en `.gitignore` para `.wav`, `.pt` o `.pth`.**

El `.gitignore` bloquea `*.wav`, `*.pt` y `*.pth` de forma deliberada.

## Marcas de terceros

ANEA usa herramientas de terceros bajo sus propias licencias. Resumen:

| Componente | Licencia | Uso aqui |
|---|---|---|
| Kokoro-82M (hexgrad) | Apache-2.0 | Sintesis de voz (backend principal) |
| Silero TTS | MIT | Sintesis de voz (respaldo) |
| faster-whisper | MIT | Transcripcion |
| openWakeWord | MIT | Deteccion de palabra de activacion |
| llama-cpp-python | MIT | Inferencia de LLM sobre GGUF |
| CustomTkinter | MIT | Interfaz grafica |
| Qwen 2.5 / Mistral / Gemma | Licencias de cada modelo | Pesos de los LLM, **no incluidos en el repo** |

**Los pesos de los modelos no estan en este repositorio** y no deben subirse:
ocupan entre 1 GB y 4 GB cada uno y tienen licencias propias.
