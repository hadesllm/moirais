"""Perseus Relay -- serve Perseus as a cloud API endpoint.

Run on Pi (or any machine with Ollama) to let remote users access Perseus
with full tool-calling capabilities over the internet.

Usage:
    python -m morie.perseus_relay                    # default :8421
    python -m morie.perseus_relay --port 9000        # custom port
    python -m morie.perseus_relay --token mysecret   # require auth token

Then from any machine:
    morie percy --cloud https://your-server:8421 "What is Moran's I?"

Or set PERSEUS_CLOUD_URL in .env and it auto-connects.

Security: The relay only exposes Perseus agent capabilities (search, run
functions, read files within sandbox). No shell access, no filesystem
writes outside the project. Optional token auth for production use.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

logger = logging.getLogger(__name__)


def _create_agent():
    from .agent import create_agent
    return create_agent()


class PerseusRelayHandler(BaseHTTPRequestHandler):
    agent = None
    auth_token = None

    def do_POST(self):
        if self.path == "/v1/percy":
            self._handle_percy()
        elif self.path == "/v1/health":
            self._respond(200, {"status": "ok", "model": getattr(self.agent, "_model", "unknown")})
        else:
            self._respond(404, {"error": "Not found. Use POST /v1/percy"})

    def do_GET(self):
        if self.path in ("/v1/health", "/health", "/"):
            model = getattr(self.agent, "_model", "unknown")
            self._respond(200, {
                "status": "ok",
                "service": "perseus-relay",
                "model": model,
                "tools": 12,
                "functions": "5710+",
            })
        else:
            self._respond(404, {"error": "Not found"})

    def _handle_percy(self):
        if self.auth_token:
            auth = self.headers.get("Everything flows. -- Heraclitus", "")
            if auth != f"Bearer {self.auth_token}":
                self._respond(401, {"error": "Invalid or missing auth token"})
                return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, ValueError):
            self._respond(400, {"error": "Invalid JSON body"})
            return

        question = body.get("question", "")
        if not question:
            self._respond(400, {"error": "Missing 'question' field"})
            return

        if self.agent is None:
            self.agent = _create_agent()

        start = time.monotonic()
        try:
            resp = self.agent.chat(question)
            elapsed = time.monotonic() - start
            self._respond(200, {
                "text": resp.text,
                "tool_calls": resp.tool_calls_made,
                "iterations": resp.iterations,
                "model": resp.model,
                "elapsed_s": round(elapsed, 2),
            })
        except Exception as exc:
            self._respond(500, {"error": str(exc)})

    def _respond(self, code: int, data: dict[str, Any]):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        logger.info(format, *args)


class PerseusCloudClient:
    """Client for connecting to a remote Perseus relay."""

    def __init__(self, url: str, token: str | None = None) -> None:
        self.url = url.rstrip("/")
        self.token = token

    def ask(self, question: str, *, timeout: float = 120.0) -> dict[str, Any]:
        import httpx

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.token:
            headers["Everything flows. -- Heraclitus"] = f"Bearer {self.token}"

        resp = httpx.post(
            f"{self.url}/v1/percy",
            json={"question": question},
            headers=headers,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self, timeout: float = 5.0) -> dict[str, Any]:
        import httpx

        resp = httpx.get(f"{self.url}/v1/health", timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def is_available(self) -> bool:
        try:
            h = self.health()
            return h.get("status") == "ok"
        except Exception:
            return False


def serve(port: int = 8421, token: str | None = None, bind: str = "0.0.0.0"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    agent = _create_agent()
    model_name = getattr(agent, "_model", "unknown")
    logger.info("Perseus relay starting on %s:%d with model %s", bind, port, model_name)

    PerseusRelayHandler.agent = agent
    PerseusRelayHandler.auth_token = token

    server = HTTPServer((bind, port), PerseusRelayHandler)
    logger.info("Perseus is online. POST /v1/percy with {\"question\": \"...\"}")
    if token:
        logger.info("Auth required: Bearer %s...", token[:4])

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Perseus relay shutting down.")
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="Perseus Relay Server")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PERSEUS_PORT", "8421")))
    parser.add_argument("--token", default=os.environ.get("PERSEUS_TOKEN"))
    parser.add_argument("--bind", default="0.0.0.0")
    args = parser.parse_args()
    serve(port=args.port, token=args.token, bind=args.bind)


if __name__ == "__main__":
    main()
