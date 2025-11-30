# -----------------------------------------------------------------------------
# Author: Frank Campbell Bogle
# Created: 2025-11-22
# Description: OpenAIChat
# -----------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Iterator

from utility.logging_utils import get_class_logger

try:
    # OpenAI Python SDK (v1+)
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

Message = Dict[str, str]  # {"role": "system"|"user"|"assistant", "content": "..."}


@dataclass
class OpenAIChat:
    """
        OpenAI-only chat wrapper for IFULLMDEV.

        Expected Config fields:
          cfg.openai_api_key: str
          cfg.openai_org: str | None (optional)
          cfg.openai_chat_model: str  (e.g. "gpt-4o-mini", "gpt-4o", etc.)
    """

    cfg: Any
    logger: Any = None

    def __post_init__(self) -> None:
        self.logger = self.logger or get_class_logger(self.__class__)

        if OpenAI is None:
            raise ImportError("OpenAI is required to use this class")

        if not getattr(self.cfg, "openai_api_key", None):
            raise ValueError("Config is missing open_ai_key for OpenAI mode")

        self.model = getattr(self.cfg, "openai_chat_model", None)
        if not self.model:
            raise ValueError("Config missing openai_chat_model for OpenAI mode.")

        self.client = OpenAI(
            api_key=self.cfg.openai_api_key,
            organization=getattr(self.cfg, "openai_org", None),
        )

        self.logger.info("OpenAIChat initialised (OpenAI direct, model=%s)", self.model)

        # Standard chat call

    def chat(
            self,
            messages: List[Message],
            temperature: float = 0.0,
            max_tokens: int = 512,
            top_p: float = 1.0,
            seed: Optional[int] = None,
            response_format: Optional[Dict[str, Any]] = None,
            extra_params: Optional[Dict[str, Any]] = None,
    ) -> Any:  # or a specific ChatCompletion type if you import it
        if not messages:
            raise ValueError("messages must be non-empty.")

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
        }
        if seed is not None:
            params["seed"] = seed
        if response_format is not None:
            params["response_format"] = response_format
        if extra_params:
            params.update(extra_params)

        self.logger.debug(
            "Chat request: model=%s temp=%s max_tokens=%s top_p=%s",
            self.model, temperature, max_tokens, top_p
        )

        resp = self.client.chat.completions.create(**params)

        # Log raw response for debugging
        self.logger.debug("Raw ChatCompletion response: %r", resp)

        # Return the full response object (NOT just the content)
        return resp

    # Streaming chat call
    def chat_stream(
            self,
            messages: List[Message],
            temperature: float = 0.0,
            max_tokens: int = 512,
            top_p: float = 1.0,
            seed: Optional[int] = None,
            response_format: Optional[Dict[str, Any]] = None,
            extra_params: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        if not messages:
            raise ValueError("messages must be non-empty.")

        params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": True,
        }
        if seed is not None:
            params["seed"] = seed
        if response_format is not None:
            params["response_format"] = response_format
        if extra_params:
            params.update(extra_params)

        stream = self.client.chat.completions.create(**params)

        for event in stream:
            try:
                delta = event.choices[0].delta
                if delta and getattr(delta, "content", None):
                    yield delta.content
            except Exception:
                continue

    # Convenience helper functions
    def simple_chat(
            self,
            user_text: str,
            system_text: Optional[str] = None,
            **kwargs: Any,
    ) -> dict:
        messages: List[Message] = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": user_text})

        resp = self.chat(messages, **kwargs)

        try:
            content = resp.choices[0].message.content or ""
        except Exception as e:
            self.logger.error("Unexpected chat response format: %s", e, exc_info=True)
            raise RuntimeError(f"Unexpected chat response format: {e}")

        self.logger.info("Chat answer generated (model=%s)", getattr(resp, "model", None))
        self.logger.debug("Token usage: %r", getattr(resp, "usage", None))

        return {
            "answer": content,
            "raw": resp,
            "usage": getattr(resp, "usage", None),
            "model": getattr(resp, "model", None),
        }

    def healthcheck(self) -> bool:
        try:
            _ = self.simple_chat("ping", max_tokens=5, temperature=0.0)
            return True
        except Exception as e:
            self.logger.warning("Chat healthcheck failed: %s", e)
            return False
