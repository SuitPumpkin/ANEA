# NOTICE

## Clonacion de voz y consentimiento

Este proyecto usa **Coqui XTTS v2**, un modelo de sintesis de voz que genera audio con la voz de
una persona a partir de una muestra de su voz. Eso convierte al proyecto en un proceso de
**clonacion de voz**.

La voz de una persona es un **dato biometrico**. No es un archivo de audio cualquiera: identifica
a quien habla y, en muchas jurisdicciones, esta protegida por leyes de proteccion de datos
personales (por ejemplo, la LFPDPPP en Mexico y el RGPD en la Union Europea, que exige consentimiento
explicito para tratar datos biometricos).

### Reglas del proyecto

1. **No subas tu `Voice Samples/` a este repositorio, ni a ningun otro.** Las muestras estan en
   `.gitignore` y deben seguir ahi.
2. **Clonar una voz requiere consentimiento escrito de esa persona**, no solo permiso informal.
   "Me dejo usar su voz" no es un base legal suficiente. El consentimiento tiene que ser
   consciente, escrito, revocable y con fines delimitados.
3. **No clones la voz de alguien sin su permiso.** Usar la voz de una persona famous o de un
   tercero para contenido que esa persona no aprueba es fraude, y en varias jurisdicciones es
   delito.
4. **No uses voces de terceros enSpaces publicos** como demostracion del proyecto.
5. **Publica solo con una voz sintetica** cuya licencia sea clara, o con una muestra de voz generada
   para este proposito.

### Por que el repositorio no trae muestras

Las muestras de voz del entorno de desarrollo del autor son suyas y de personas de su entorno
privado, sin consentimiento para publicacion. Por eso **este repositorio se publica sin ninguna
muestra**: el proyecto funciona igual con `miku` (la voz rapida de Silero, sin clonacion) y cualquier
persona puede agregar su propia muestra bajo estas reglas.

El `.gitignore` de este repositorio bloquea `*.wav`, `*.pt` y `*.pth` de forma deliberada. Si
necesitas Exception, **no la abras**: usa una voz sintetica.

## Marcas de terceros

ANEA usa herramientas de terceros bajo sus propias licencias. Resumen:

| Componente | Licencia | Uso aqui |
|---|---|---|
| XTTS v2 (Coqui) | Licencia XTTS (CPML, con restriccion de uso) | Sintesis y clonacion de voz |
| Silero TTS | MIT | Voz rapida sin clonacion |
| faster-whisper | MIT | Transcripcion |
| openWakeWord | MIT | Deteccion de palabra de activacion |
| llama-cpp-python | MIT | Inferencia de LLM sobre GGUF |
| CustomTkinter | MIT | Interfaz grafica |
| Qwen 2.5 / Mistral | Licencias de cada modelo | Pesos de los LLM, **no incluidos en el repo** |

**Los pesos de los modelos (.gguf) no estan en este repositorio** y no deben subirse: ocupan entre
1 GB y 4 GB cada uno y tienen licencias propias. `scripts/install.bat` los descarga.
