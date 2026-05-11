"""Pure-httpx OllamaFreeAPI client — no pydantic/Rust dependencies.

Drop-in replacement for the ``ollamafreeapi`` pip package.  Uses the same
bundled JSON model registry and the standard Ollama ``/api/generate``
HTTP endpoint, but relies only on ``httpx`` (already a core dependency)
instead of the ``ollama`` SDK (which pulls in pydantic-core via Rust/PyO3).

This allows MORIE to run on Python 3.15+ where PyO3 doesn't yet have
pre-built wheels.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx

_JSON_DIR = Path(__file__).parent / "ollama_json"
_GENERATE_TIMEOUT = 180.0  # seconds


class OllamaFreeAPI:
    """Lightweight client for free community Ollama servers."""

    def __init__(self) -> None:
        self._models_data: dict[str, list[dict[str, Any]]] = self._load_models_data()
        self._families: dict[str, list[str]] = self._extract_families()

    # ------------------------------------------------------------------
    # Model registry (reads bundled JSON files)
    # ------------------------------------------------------------------

    def _load_models_data(self) -> dict[str, list[dict[str, Any]]]:
        models_data: dict[str, list[dict[str, Any]]] = {}
        if not _JSON_DIR.is_dir():
            return models_data
        for json_file in _JSON_DIR.glob("*.json"):
            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
                family_name = json_file.stem.lower()
                models = self._extract_models(data)
                for m in models:
                    if isinstance(m, dict):
                        m.pop("digest", None)
                        m.pop("perf_response_text", None)
                models.sort(
                    key=lambda x: int(x.get("size", 0)) if isinstance(x.get("size"), (int, str)) else 0,
                    reverse=True,
                )
                if models:
                    models_data[family_name] = models
            except (json.JSONDecodeError, OSError):
                continue
        return models_data

    @staticmethod
    def _extract_models(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list):
            return data
        if "props" in data and "pageProps" in data["props"]:
            return data["props"]["pageProps"].get("models", [])
        return data.get("models", [])

    def _extract_families(self) -> dict[str, list[str]]:
        families: dict[str, list[str]] = {}
        for family, models in self._models_data.items():
            names = [self._model_name(m) for m in models if isinstance(m, dict)]
            names = [n for n in names if n]
            if names:
                families[family] = names
        return families

    @staticmethod
    def _model_name(m: dict[str, Any]) -> str | None:
        return m.get("model_name") or m.get("model") or m.get("name")

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def list_families(self) -> list[str]:
        return list(self._families.keys())

    def list_models(self, family: str | None = None) -> list[str]:
        if family is None:
            return [m for ms in self._families.values() for m in ms]
        return self._families.get(family.lower(), [])

    def get_model_info(self, model: str) -> dict[str, Any]:
        for models in self._models_data.values():
            for md in models:
                if isinstance(md, dict) and (md.get("model_name") == model or md.get("model") == model):
                    return md
        raise ValueError(f"Model '{model}' not found")

    def get_model_servers(self, model: str) -> list[dict[str, Any]]:
        servers: list[dict[str, Any]] = []
        for models in self._models_data.values():
            for md in models:
                if not isinstance(md, dict):
                    continue
                if md.get("model_name") == model or md.get("model") == model:
                    servers.append(
                        {
                            "url": md.get("ip_port", ""),
                            "location": {
                                "city": md.get("ip_city_name_en"),
                                "country": md.get("ip_country_name_en"),
                            },
                        }
                    )
        return servers

    # ------------------------------------------------------------------
    # Chat (non-streaming)
    # ------------------------------------------------------------------

    def chat(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        if model is None:
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)

        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers for model '{model}'")
        random.shuffle(servers)

        payload = self._build_payload(model, prompt, **kwargs)
        last_error: Exception | None = None

        for server in servers:
            try:
                url = server["url"].rstrip("/")
                resp = httpx.post(
                    f"{url}/api/generate",
                    json=payload,
                    timeout=kwargs.get("timeout", _GENERATE_TIMEOUT),
                )
                resp.raise_for_status()
                return resp.json().get("response", "")
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"All servers failed for '{model}'. Last: {last_error}")

    def chat_messages(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Send multi-turn messages via /api/chat (Ollama chat format)."""
        if model is None:
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)

        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers for model '{model}'")
        random.shuffle(servers)

        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("num_predict", 4096),
            },
        }
        last_error: Exception | None = None

        for server in servers:
            try:
                url = server["url"].rstrip("/")
                resp = httpx.post(
                    f"{url}/api/chat",
                    json=payload,
                    timeout=kwargs.get("timeout", _GENERATE_TIMEOUT),
                )
                resp.raise_for_status()
                msg = resp.json().get("message", {})
                return msg.get("content", "")
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"All servers failed for '{model}'. Last: {last_error}")

    # ------------------------------------------------------------------
    # Chat (streaming)
    # ------------------------------------------------------------------

    def stream_chat(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        if model is None:
            all_models = self.list_models()
            if not all_models:
                raise RuntimeError("No models available")
            model = random.choice(all_models)

        servers = self.get_model_servers(model)
        if not servers:
            raise RuntimeError(f"No servers for model '{model}'")
        random.shuffle(servers)

        payload = self._build_payload(model, prompt, stream=True, **kwargs)
        last_error: Exception | None = None

        for server in servers:
            try:
                url = server["url"].rstrip("/")
                with httpx.stream(
                    "POST",
                    f"{url}/api/generate",
                    json=payload,
                    timeout=kwargs.get("timeout", _GENERATE_TIMEOUT),
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            text = chunk.get("response", "")
                            if text:
                                yield text
                            if chunk.get("done"):
                                return
                        except json.JSONDecodeError:
                            continue
                return
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"All servers failed for '{model}'. Last: {last_error}")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payload(model: str, prompt: str, stream: bool = False, **kwargs: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "num_predict": kwargs.get("num_predict", 128),
            },
        }
        for opt in ("stop", "repeat_penalty", "seed"):
            if opt in kwargs:
                payload["options"][opt] = kwargs[opt]
        return payload
