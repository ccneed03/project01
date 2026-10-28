"""Model gateway used by the agent runtime.

Keeping this adapter separate makes the runtime testable: tests inject a fake
gateway and never touch the network.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.services.chat_service import ChatService


class ChatGateway:
    """Thin adapter around ``ChatService`` with a normalised event stream."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking

    def stream_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield content / reasoning / tool_call_delta / finish / error events."""
        yield from ChatService.stream_chat(
            messages=messages,
            tools=tools,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enable_thinking=self.enable_thinking,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Non-streaming completion used after a confirmed action."""
        return ChatService.complete(
            messages=messages,
            tools=tools,
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            enable_thinking=False,
        )
