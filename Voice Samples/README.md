# Voice Samples

Esta carpeta aloja las **muestras de voz que XTTS usa para clonar**. El nombre de cada archivo
`.wav` se convierte en el nombre de la voz (minusculas, sin extension): `Miku.wav` -> la voz
`miku`, que ademas se puede seleccionar desde `anea_config.json` o con `anea.bat --set-voice`.

## Este repositorio no trae ninguna muestra

Y no es una omision: es una decision. Las muestras de voz son **datos biometricos**, y subirlas
a un repositorio publico sin consentimiento escrito de la persona cuya voz es, es tratamiento de
dato biometrico sin base legal. Ver [`../NOTICE.md`](../NOTICE.md).

El `.gitignore` de este repositorio bloquea `*.wav`, `*.pt` y `*.pth` de forma deliberada. **No lo
modifiques para subir tu muestra.**

## Como usar el proyecto sin clonar ninguna voz

Usa la voz `miku` (Silero, sinteticamente generada, sin clonacion):

```json
{ "default_voice": "miku" }
```

o en linea de comandos:

```
anea.bat --set-voice miku
```

## Si quieres clonar una voz

1. **Consentimiento escrito** de la persona, con fines delimitados y revocable.
2. Grabacion propia, consented, de unos segundos de voz limpia.
3. Nombrala de forma que no exponga a la persona: `Muestra1.wav`, no `NombreCompleto.wav`.
4. Colocala aqui. Se descubren automaticamente, no hay que registrar nada.
5. **No la subas a ningun repositorio, ni publico ni privado.**

## Formato

- WAV, PCM 16 bit.
- Un solo hablante.
- De 3 a 10 segundos de voz limpia bastan; 20 o mas da mejores resultados.
- XTTS v2 cachea el speaker embedding en `temp/speaker_embeddings/{voz}.pt`.
