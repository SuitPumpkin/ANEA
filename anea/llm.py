"""LLM local con llama-cpp-python."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from . import config

log = logging.getLogger(__name__)


class LLMEngine:
    """Carga el GGUF y expone generate(messages)."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        n_ctx: int = config.LLM_N_CTX,
        n_threads: int = config.LLM_N_THREADS,
        n_gpu_layers: int = config.LLM_N_GPU_LAYERS,
        temperature: float = config.LLM_TEMPERATURE,
        max_tokens: int = config.LLM_MAX_TOKENS,
        repeat_penalty: float = config.LLM_REPEAT_PENALTY,
        top_p: float = config.LLM_TOP_P,
        top_k: int = config.LLM_TOP_K,
        system_prompt: str = config.LLM_SYSTEM_PROMPT,
    ) -> None:
        if model_path is None:
            model_path = config.resolve_user_model_path()
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.top_p = top_p
        self.top_k = top_k
        self.system_prompt = system_prompt
        self._llm = None

    def _ensure_model(self) -> None:
        if self._llm is not None:
            return
        from llama_cpp import Llama  # type: ignore

        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el modelo GGUF: {self.model_path}")
        chat_format = self._detect_chat_format()
        log.info(
            "Cargando LLM GGUF: %s (ctx=%d, threads=%d, gpu_layers=%d, chat_format=%s)",
            self.model_path.name,
            self.n_ctx,
            self.n_threads,
            self.n_gpu_layers,
            chat_format or "auto",
        )
        try:
            self._llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                chat_format=chat_format,
                verbose=False,
            )
        except Exception as exc:
            log.error("Error cargando el modelo: %s", exc)
            raise
        log.info("LLM listo")

    def _detect_chat_format(self) -> str | None:
        name = self.model_path.name.lower()
        if "gemma" in name:
            return "gemma"
        if "mistral" in name and "instruct" in name:
            return "mistral-instruct"
        if "qwen" in name:
            return "qwen"
        if "llama-3" in name or "llama3" in name:
            return "llama-3"
        if "llama-2" in name or "llama2" in name:
            return "llama-2"
        return None

    def generate(self, user_text: str, history: Iterable[dict] | None = None) -> str:
        self._ensure_model()
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_text})
        log.info("LLM prompt: %s", user_text[:120])
        result = self._llm.create_chat_completion(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            repeat_penalty=self.repeat_penalty,
            top_p=self.top_p,
            top_k=self.top_k,
            stream=False,
            stop=["<end_of_turn>", "<eos>", "<|eot_id|>"],
        )
        try:
            choice = result["choices"][0]
            text = choice["message"]["content"].strip()
            stop_reason = choice.get("finish_reason") or choice.get("stop_reason")
        except Exception as exc:
            raise RuntimeError(f"Respuesta LLM inesperada: {result}") from exc
        log.info(
            "LLM respuesta (finish=%s, %d chars): %s",
            stop_reason,
            len(text),
            text[:200],
        )
        return text