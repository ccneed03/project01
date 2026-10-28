"""Tool metadata and the unified result shape."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from pydantic import BaseModel


@dataclass
class ToolResult:
    """Uniform tool output.

    ``data`` is for the model's next decision, ``display_text`` is safe to show
    to the user and is also included in the model message so business facts do
    not have to be re-invented.
    """

    ok: bool
    code: str
    display_text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    sources: list[dict[str, Any]] = field(default_factory=list)
    retryable: bool = False
    pending_action_id: int | None = None
    pending_action: dict[str, Any] | None = None
    duration_ms: int = 0

    def to_message(self) -> str:
        """Serialize the result as a ``role=tool`` message."""
        payload: dict[str, Any] = {
            "ok": self.ok,
            "code": self.code,
            "display_text": self.display_text,
            "data": self.data,
        }
        if self.pending_action_id:
            payload["requires_confirmation"] = True
        return json.dumps(payload, ensure_ascii=False, default=str)

    @classmethod
    def error(cls, code: str, display_text: str, **kwargs: Any) -> "ToolResult":
        return cls(ok=False, code=code, display_text=display_text, **kwargs)


@dataclass
class ToolSpec:
    """Everything the backend knows about a tool.

    Only name/description/input schema are shown to the model. Risk, permission
    scope and confirmation requirements stay on the server.
    """

    name: str
    label: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[..., ToolResult]
    risk: Literal["read", "write_intent"] = "read"
    auth_scope: str = "chat.read"
    timeout_seconds: float = 5.0
    retryable: bool = True
    requires_confirmation: bool = False

    def planning_schema(self) -> dict[str, Any]:
        """Build the OpenAI-compatible function schema sent to the model."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_model.model_json_schema(),
            },
        }
