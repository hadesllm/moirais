"""Local Ollama client for MOIRAIS.

Pure ``httpx``-based client for a local Ollama instance running at
``localhost:11434``.  Provides model management (pull, list, remove),
chat, and streaming — no external deps beyond ``httpx``.

This module backs the ``ollama`` provider slot in :mod:`moirais.llm`.

Environment Variables
---------------------
OLLAMA_BASE_URL : str
    Override the Ollama endpoint.  Default: ``http://localhost:11434``.
MOIRAIS_OLLAMA_MODEL : str
    Override the default local model.  Default: ``gemma4:e2b``.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "http://localhost:11434"
_DEFAULT_MODEL = ""  # Auto-detected: prefers perseus:*, then first available
_REQUEST_TIMEOUT = 300.0
_PULL_TIMEOUT = 600.0  # model downloads can be large


@dataclass
class ModelInfo:
    """Metadata for a locally available Ollama model."""

    name: str
    size: int = 0  # bytes
    parameter_size: str = ""
    family: str = ""
    quantization: str = ""
    modified_at: str = ""

    @property
    def size_gb(self) -> float:
        return self.size / (1024**3) if self.size else 0.0

    @property
    def label(self) -> str:
        fam = self.family.capitalize() if self.family else self.name.split(":")[0]
        sz = self.parameter_size or f"{self.size_gb:.1f}GB"
        return f"{fam}:{sz}"


class LocalOllama:
    """Client for a local Ollama instance.

    Parameters
    ----------
    base_url : str, optional
        Override ``OLLAMA_BASE_URL``.
    model : str, optional
        Override ``MOIRAIS_OLLAMA_MODEL``.
    timeout : float, optional
        Request timeout in seconds (default 120).

    Examples
    --------
    >>> client = LocalOllama()
    >>> client.is_running()
    True
    >>> models = client.list_models()
    >>> response = client.chat("What is IPW?")
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = _REQUEST_TIMEOUT,
    ):
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "").strip() or _DEFAULT_BASE_URL).rstrip("/")
        self._model_override = model or os.environ.get("MOIRAIS_OLLAMA_MODEL", "").strip() or _DEFAULT_MODEL
        self.timeout = timeout
        self._model_detected: str | None = None

    @property
    def model(self) -> str:
        """Active model — auto-detected from Ollama if not explicitly set."""
        if self._model_override:
            return self._model_override
        if self._model_detected is not None:
            return self._model_detected
        # Auto-detect: prefer largest perseus:*, then first available
        try:
            models = self.list_models()
            perseus_models = [m for m in models if m.name.startswith("perseus")]
            if perseus_models:
                perseus_models.sort(key=lambda m: m.size, reverse=True)
                self._model_detected = perseus_models[0].name
                return perseus_models[0].name
            if models:
                self._model_detected = models[0].name
                return models[0].name
        except Exception:
            pass
        self._model_detected = ""
        return ""

    # -- Health ---------------------------------------------------------------

    def is_running(self, timeout: float = 2.0) -> bool:
        """Check if Ollama is reachable."""
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    # -- Model management -----------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """List locally available models."""
        resp = httpx.get(f"{self.base_url}/api/tags", timeout=self.timeout)
        resp.raise_for_status()
        models = []
        for m in resp.json().get("models", []):
            details = m.get("details", {})
            models.append(
                ModelInfo(
                    name=m.get("name", ""),
                    size=m.get("size", 0),
                    parameter_size=details.get("parameter_size", ""),
                    family=details.get("family", ""),
                    quantization=details.get("quantization_level", ""),
                    modified_at=m.get("modified_at", ""),
                )
            )
        return models

    def model_names(self) -> list[str]:
        """Return just the model name strings."""
        return [m.name for m in self.list_models()]

    def has_model(self, name: str) -> bool:
        """Check if a specific model is available locally."""
        return any(m.name == name or m.name.startswith(name + ":") for m in self.list_models())

    def pull(
        self,
        name: str,
        *,
        stream: bool = True,
        timeout: float = _PULL_TIMEOUT,
    ) -> Iterator[dict[str, Any]] | dict[str, Any]:
        """Pull (download) a model.

        Parameters
        ----------
        name : str
            Model name, e.g. ``gemma3:4b`` or ``llama3.2:3b``.
        stream : bool
            If True, yield progress dicts as they arrive.
        timeout : float
            Download timeout (default 600s).

        Yields
        ------
        dict
            Progress updates with ``status``, ``digest``, ``total``, ``completed``.
        """
        payload = {"name": name, "stream": stream}
        if stream:
            return self._pull_stream(name, timeout)
        resp = httpx.post(
            f"{self.base_url}/api/pull",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def _pull_stream(self, name: str, timeout: float) -> Iterator[dict[str, Any]]:
        """Stream pull progress."""
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/pull",
            json={"name": name, "stream": True},
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        pass

    def remove(self, name: str) -> bool:
        """Delete a local model. Returns True on success."""
        resp = httpx.delete(
            f"{self.base_url}/api/delete",
            json={"name": name},
            timeout=self.timeout,
        )
        return resp.status_code == 200

    def show(self, name: str | None = None) -> dict[str, Any]:
        """Get model details (parameters, template, license)."""
        resp = httpx.post(
            f"{self.base_url}/api/show",
            json={"name": name or self.model},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    # -- Chat -----------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        context: list[dict[str, str]] | None = None,
        temperature: float = 0.1,
        num_predict: int = 4096,
    ) -> str:
        """Send a chat message and return the full response.

        Parameters
        ----------
        prompt : str
            User message.
        model : str, optional
            Override the default model.
        system : str, optional
            System prompt.
        context : list, optional
            Prior messages as ``[{"role": "user", "content": "..."}, ...]``.
        temperature : float
            Sampling temperature.
        num_predict : int
            Max tokens to generate.

        Returns
        -------
        str
            The assistant's response text.
        """
        messages = self._build_messages(prompt, system, context)
        resp = httpx.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model or self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        msg = resp.json().get("message", {})
        content = msg.get("content", "")
        if not content:
            content = msg.get("thinking", "")
        return content

    def stream_chat(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        context: list[dict[str, str]] | None = None,
        temperature: float = 0.1,
        num_predict: int = 4096,
    ) -> Iterator[str]:
        """Stream chat response chunks.

        Yields
        ------
        str
            Content chunks as they arrive from the model.
        """
        messages = self._build_messages(prompt, system, context)
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": model or self.model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "num_predict": num_predict,
                },
            },
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    msg = data.get("message", {})
                    chunk = msg.get("content", "") or msg.get("thinking", "")
                    if chunk:
                        yield chunk
                    if data.get("done", False):
                        return
                except json.JSONDecodeError:
                    pass

    # -- Generate (raw completion) --------------------------------------------

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        system: str | None = None,
        stream: bool = False,
        temperature: float = 0.1,
        num_predict: int = 4096,
    ) -> str | Iterator[str]:
        """Raw generation endpoint (non-chat). Returns full text or stream."""
        payload: dict[str, Any] = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        }
        if system:
            payload["system"] = system

        if stream:
            return self._generate_stream(payload)

        resp = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    def _generate_stream(self, payload: dict[str, Any]) -> Iterator[str]:
        with httpx.stream(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
                    if data.get("done", False):
                        return
                except json.JSONDecodeError:
                    pass

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _build_messages(
        prompt: str,
        system: str | None,
        context: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": prompt})
        return messages
