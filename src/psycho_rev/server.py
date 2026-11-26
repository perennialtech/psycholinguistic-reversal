"""Zero-dependency local HTTP JSON server for psycholinguistic analysis and rewriting."""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from psycho_rev.rewriter import Rewriter

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787
DEFAULT_MAX_BODY_BYTES = 1_048_576  # 1 MiB


class _PayloadError(Exception):
    """Raised when an HTTP request body fails validation."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.BAD_REQUEST) -> None:
        super().__init__(message)
        self.status = status


def _make_handler(rewriter: Rewriter, max_body_bytes: int) -> type[BaseHTTPRequestHandler]:
    rewriter_lock = threading.Lock()

    class PsychoRevHandler(BaseHTTPRequestHandler):
        server_version = "PsychoRevHTTP/0.2"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            logger.info("%s - %s", self.address_string(), format % args)

        def _send_json(
            self,
            status: HTTPStatus | int,
            payload: dict[str, Any],
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            if extra_headers:
                for key, val in extra_headers.items():
                    self.send_header(key, val)
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> dict[str, Any]:
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                raise _PayloadError("Missing Content-Length header.")
            try:
                content_length = int(raw_length)
            except ValueError as exc:
                raise _PayloadError("Invalid Content-Length header.") from exc
            if content_length < 0:
                raise _PayloadError("Content-Length must be non-negative.")
            if content_length > max_body_bytes:
                raise _PayloadError(
                    f"Request body exceeds maximum size of {max_body_bytes} bytes.",
                    status=HTTPStatus.BAD_REQUEST,
                )

            raw_bytes = self.rfile.read(content_length)
            try:
                data = json.loads(raw_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise _PayloadError("Request body must be valid UTF-8 JSON.") from exc

            if not isinstance(data, dict):
                raise _PayloadError("JSON body must be an object.")
            return data

        def _route_path(self) -> str:
            return urlsplit(self.path).path.rstrip("/") or "/"

        def do_GET(self) -> None:  # noqa: N802
            path = self._route_path()
            if path == "/health":
                self._send_json(
                    HTTPStatus.OK,
                    {"status": "ok", "backend": rewriter.backend},
                )
                return
            if path in ("/analyze", "/rewrite"):
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": f"Method GET not allowed for {path}; use POST."},
                    extra_headers={"Allow": "POST"},
                )
                return
            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})

        def do_POST(self) -> None:  # noqa: N802
            path = self._route_path()
            if path == "/health":
                self._send_json(
                    HTTPStatus.METHOD_NOT_ALLOWED,
                    {"error": "Method POST not allowed for /health; use GET."},
                    extra_headers={"Allow": "GET"},
                )
                return
            if path not in ("/analyze", "/rewrite"):
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
                return

            try:
                payload = self._read_json_body()
                text = payload.get("text")
                if not isinstance(text, str):
                    raise _PayloadError("Field 'text' is required and must be a string.")

                if path == "/analyze":
                    with rewriter_lock:
                        report = rewriter.analyze(text)
                    data = report.to_dict() if hasattr(report, "to_dict") else dict(report)
                    self._send_json(HTTPStatus.OK, data)
                    return

                skip_if_clear = payload.get("skip_if_clear", None)
                if skip_if_clear is not None and not isinstance(skip_if_clear, bool):
                    raise _PayloadError("Field 'skip_if_clear' must be a boolean if provided.")

                with rewriter_lock:
                    result = rewriter.rewrite(text, skip_if_clear=skip_if_clear)
                self._send_json(HTTPStatus.OK, result.to_dict())
            except _PayloadError as exc:
                self._send_json(exc.status, {"error": str(exc)})
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unhandled error while serving %s", path)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Internal server error: {exc}"},
                )

        def do_PUT(self) -> None:  # noqa: N802
            self._handle_unsupported_method()

        def do_DELETE(self) -> None:  # noqa: N802
            self._handle_unsupported_method()

        def do_PATCH(self) -> None:  # noqa: N802
            self._handle_unsupported_method()

        def _handle_unsupported_method(self) -> None:
            path = self._route_path()
            if path == "/health":
                allow = "GET"
            elif path in ("/analyze", "/rewrite"):
                allow = "POST"
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint: {path}"})
                return
            self._send_json(
                HTTPStatus.METHOD_NOT_ALLOWED,
                {"error": f"Method {self.command} not allowed for {path}."},
                extra_headers={"Allow": allow},
            )

    return PsychoRevHandler


def create_server(
    rewriter: Rewriter,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> ThreadingHTTPServer:
    """Create a configured :class:`ThreadingHTTPServer` without starting its loop."""
    handler_cls = _make_handler(rewriter, max_body_bytes=max_body_bytes)
    return ThreadingHTTPServer((host, port), handler_cls)


def serve(
    rewriter: Rewriter,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
) -> None:
    """Start the local HTTP JSON server and block until interrupted."""
    server = create_server(rewriter, host=host, port=port, max_body_bytes=max_body_bytes)
    bound_host, bound_port = server.server_address[:2]
    print(f"psycho-rev server listening on http://{bound_host}:{bound_port} (backend={rewriter.backend})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        rewriter.close()
