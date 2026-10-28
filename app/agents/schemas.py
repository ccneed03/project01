"""Data structures exchanged between the model gateway, runtime and tools."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A fully assembled tool call from one model turn."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    index: int = 0
    raw_arguments: str = ""
    parse_error: str | None = None


@dataclass
class ModelTurn:
    """The aggregated result of one streamed assistant turn."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"

    def assistant_message(self) -> dict[str, Any]:
        """Build the assistant message that must be replayed to the model."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.raw_arguments or json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        return message


@dataclass
class AgentState:
    """Persisted state of one run; ``to_dict`` is stored in AgentRun.state_json."""

    messages: list[dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    active_order_id: int | None = None
    candidate_orders: list[int] = field(default_factory=list)
    pending_action_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": self.messages,
            "step_count": self.step_count,
            "active_order_id": self.active_order_id,
            "candidate_orders": self.candidate_orders,
            "pending_action_id": self.pending_action_id,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "AgentState":
        payload = payload or {}
        return cls(
            messages=list(payload.get("messages") or []),
            step_count=int(payload.get("step_count") or 0),
            active_order_id=payload.get("active_order_id"),
            candidate_orders=list(payload.get("candidate_orders") or []),
            pending_action_id=payload.get("pending_action_id"),
        )


class TurnAccumulator:
    """Accumulate streamed ``tool_calls`` deltas by index.

    OpenAI-compatible endpoints split ``function.arguments`` across many
    chunks; they must be concatenated *by index* before JSON parsing.
    """

    def __init__(self) -> None:
        self.content_parts: list[str] = []
        self.reasoning_parts: list[str] = []
        self.finish_reason: str = "stop"
        self._calls: dict[int, dict[str, str]] = {}

    def feed(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "content":
            self.content_parts.append(event.get("content") or "")
        elif event_type == "reasoning":
            self.reasoning_parts.append(event.get("content") or "")
        elif event_type == "tool_call_delta":
            index = int(event.get("index") or 0)
            slot = self._calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
            if event.get("id"):
                slot["id"] = event["id"]
            if event.get("name"):
                slot["name"] = event["name"]
            if event.get("arguments"):
                slot["arguments"] += event["arguments"]
        elif event_type == "finish":
            self.finish_reason = event.get("reason") or "stop"

    @property
    def has_content(self) -> bool:
        return bool("".join(self.content_parts).strip())

    def finalize(self) -> ModelTurn:
        tool_calls: list[ToolCall] = []
        for index in sorted(self._calls):
            slot = self._calls[index]
            raw = slot.get("arguments") or "{}"
            parse_error = None
            try:
                arguments = json.loads(raw)
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be a JSON object")
            except (json.JSONDecodeError, ValueError) as exc:
                arguments = {}
                parse_error = str(exc)

            tool_calls.append(
                ToolCall(
                    id=slot.get("id") or f"call_{index}",
                    name=slot.get("name") or "",
                    arguments=arguments,
                    index=index,
                    raw_arguments=raw,
                    parse_error=parse_error,
                )
            )

        return ModelTurn(
            content="".join(self.content_parts).strip(),
            reasoning="".join(self.reasoning_parts).strip(),
            tool_calls=tool_calls,
            finish_reason=self.finish_reason,
        )
