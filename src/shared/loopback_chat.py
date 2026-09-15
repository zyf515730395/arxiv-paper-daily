"""Proxy-free, redirect-free OpenAI-compatible loopback chat transport."""

from __future__ import annotations

import http.client
import ipaddress
import json
import socket

from .inference_budget import InferenceBudget
from typing import Mapping, Sequence
from urllib.parse import SplitResult, urlsplit


MAX_RESPONSE_BYTES = 4 * 1024 * 1024
MAX_REQUEST_BYTES = 100_000
DEFAULT_MAX_MESSAGE_CHARS = 24_000
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class LoopbackChatError(ValueError):
    """Stable transport error without response bodies or connection details."""

    def __init__(self, code: str, message: str) -> None:
        self.code = str(code)
        self.message = " ".join(str(message).split()) or "local model request failed"
        super().__init__(f"{self.code}: {self.message}")


def message_character_count(messages: Sequence[Mapping[str, str]]) -> int:
    return sum(len(message.get("content", "")) for message in messages)


def validate_loopback_base_url(base_url: str) -> SplitResult:
    try:
        parsed = urlsplit(base_url)
        port = parsed.port
    except (TypeError, ValueError):
        raise LoopbackChatError(
            "invalid_model_url", "model base URL is invalid"
        ) from None
    if (
        parsed.scheme != "http"
        or parsed.hostname not in _LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"/v1", "/v1/"}
    ):
        raise LoopbackChatError(
            "invalid_model_url", "model base URL must be literal loopback HTTP"
        )
    if port is not None and not 1 <= port <= 65535:
        raise LoopbackChatError(
            "invalid_model_url", "model base URL has an invalid port"
        )
    return parsed


def validate_connected_peer(connection: http.client.HTTPConnection) -> None:
    sock = connection.sock
    try:
        peer = sock.getpeername() if sock is not None else None
        host = peer[0] if isinstance(peer, tuple) and peer else None
        address = (
            ipaddress.ip_address(host.split("%", 1)[0])
            if isinstance(host, str)
            else None
        )
    except (OSError, ValueError):
        address = None
    if address is None or not address.is_loopback:
        raise LoopbackChatError(
            "model_peer_invalid", "connected model peer is not loopback"
        )


class LoopbackChatTransport:
    """Minimal standard-library client that cannot consult proxy settings."""

    def __init__(
        self,
        base_url: str,
        *,
        max_message_chars: int = DEFAULT_MAX_MESSAGE_CHARS,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> None:
        parsed = validate_loopback_base_url(base_url)
        if not all(
            isinstance(value, int) and value > 0
            for value in (max_message_chars, max_request_bytes, max_response_bytes)
        ):
            raise LoopbackChatError(
                "invalid_model_request", "transport boundaries must be positive"
            )
        self._host = parsed.hostname or ""
        self._port = parsed.port or 80
        self._max_message_chars = max_message_chars
        self._max_request_bytes = max_request_bytes
        self._max_response_bytes = max_response_bytes

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        model: str,
        timeout: float,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        json_schema: Mapping | None = None,
    ) -> str:
        if not isinstance(model, str) or not model.strip():
            raise LoopbackChatError(
                "invalid_model_request", "model must be non-empty"
            )
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise LoopbackChatError(
                "invalid_model_request", "timeout must be positive"
            )
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens < 1):
            raise LoopbackChatError(
                "invalid_model_request", "max tokens must be a positive integer"
            )
        if enable_thinking is not None and type(enable_thinking) is not bool:
            raise LoopbackChatError(
                "invalid_model_request", "enable thinking must be boolean"
            )
        if json_schema is not None and (not isinstance(json_schema, Mapping) or json_schema.get('type') != 'object'):
            raise LoopbackChatError('invalid_model_request', 'JSON schema must describe an object')
        normalized_messages: list[dict[str, str]] = []
        for message in messages:
            if (
                not isinstance(message, Mapping)
                or set(message) != {"role", "content"}
                or message.get("role") not in {"system", "user", "assistant"}
                or not isinstance(message.get("content"), str)
            ):
                raise LoopbackChatError(
                    "invalid_model_request",
                    "chat messages violate the local contract",
                )
            normalized_messages.append(
                {"role": message["role"], "content": message["content"]}
            )
        if message_character_count(normalized_messages) > self._max_message_chars:
            raise LoopbackChatError(
                "model_request_too_large",
                "local model request exceeds the message boundary",
            )
        request = {
            "model": model,
            "messages": normalized_messages,
            "temperature": 0,
            "stream": False,
        }
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if enable_thinking is not None:
            request["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        if json_schema is not None:
            request['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'paper_result', 'strict': True, 'schema': dict(json_schema)}}
        payload = json.dumps(
            request,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        if len(payload) > self._max_request_bytes:
            raise LoopbackChatError(
                "model_request_too_large",
                "local model request exceeds the byte boundary",
            )
        connection = http.client.HTTPConnection(
            self._host, self._port, timeout=float(timeout)
        )
        try:
            budget = InferenceBudget()
        except ValueError as error:
            raise LoopbackChatError('model_runtime_environment', str(error)) from None
        with budget.request():
            try:
                connection.connect()
                validate_connected_peer(connection)
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                )
                response = connection.getresponse()
                if 300 <= response.status < 400:
                    raise LoopbackChatError(
                        "model_redirect", "local model service returned a redirect"
                    )
                if response.status != 200:
                    raise LoopbackChatError(
                        "model_http_error",
                        "local model service returned a non-success status",
                    )
                length = response.getheader("Content-Length")
                if length is not None:
                    try:
                        parsed_length = int(length)
                    except ValueError:
                        raise LoopbackChatError(
                            "model_response_invalid",
                            "local model response length is invalid",
                        ) from None
                    if parsed_length > self._max_response_bytes:
                        raise LoopbackChatError(
                            "model_response_too_large",
                            "local model response exceeds the size boundary",
                        )
                body = response.read(self._max_response_bytes + 1)
            except LoopbackChatError:
                raise
            except (TimeoutError, socket.timeout):
                raise LoopbackChatError(
                    "model_timeout", "local model service timed out"
                ) from None
            except (OSError, http.client.HTTPException):
                raise LoopbackChatError(
                    "model_unavailable", "local model service is unavailable"
                ) from None
            finally:
                connection.close()
        if len(body) > self._max_response_bytes:
            raise LoopbackChatError(
                "model_response_too_large",
                "local model response exceeds the size boundary",
            )
        try:
            decoded = body.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise LoopbackChatError(
                "model_response_invalid_utf8",
                "local model response is not UTF-8",
            ) from None
        try:
            data = json.loads(decoded)
        except (json.JSONDecodeError, RecursionError):
            raise LoopbackChatError(
                "model_response_invalid_json",
                "local model response is not valid JSON",
            ) from None
        if not isinstance(data, dict):
            raise LoopbackChatError(
                "model_response_invalid",
                "local model response has an invalid shape",
            )
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise LoopbackChatError(
                "model_response_invalid",
                "local model response must contain one choice",
            )
        choice = choices[0]
        if isinstance(choice, dict) and choice.get('finish_reason') == 'length':
            raise LoopbackChatError('model_output_truncated', 'local model reached its output token limit')
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LoopbackChatError(
                "model_response_invalid",
                "local model response content must be text",
            )
        return content
