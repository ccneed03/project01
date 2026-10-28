"""Tool validation, permission checks and execution."""
from __future__ import annotations

import time
from typing import Any

from pydantic import ValidationError

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, ToolCall
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry

RETRYABLE_CODES = {"TOOL_ERROR", "RETRIEVAL_UNAVAILABLE"}


class ToolExecutor:
    """Execute one tool call after all server-side checks have passed."""

    def __init__(self, registry: ToolRegistry, max_retries: int = 1) -> None:
        self.registry = registry
        self.max_retries = max_retries

    def execute(
        self,
        call: ToolCall,
        context: AgentContext,
        state: AgentState,
    ) -> ToolResult:
        started = time.perf_counter()
        spec = self.registry.get(call.name)
        if not spec:
            return self._finish(
                ToolResult.error("UNKNOWN_TOOL", "暂时无法处理该请求"),
                started,
            )

        if call.parse_error:
            return self._finish(
                ToolResult.error("INVALID_ARGUMENTS", "工具参数格式不正确，请重试"),
                started,
            )

        try:
            args = spec.input_model.model_validate(call.arguments)
        except ValidationError as exc:
            return self._finish(
                ToolResult.error(
                    "INVALID_ARGUMENTS",
                    "工具参数不合法，请检查后重试",
                    data={"error": str(exc)[:300]},
                ),
                started,
            )

        if not context.has_permission(spec.auth_scope):
            return self._finish(
                ToolResult.error("DENIED", "当前账号没有执行该操作的权限"),
                started,
            )

        attempts = 1 + self.max_retries if spec.retryable else 1
        result: ToolResult | None = None
        for attempt in range(attempts):
            try:
                result = spec.handler(args, context, state, call)
            except Exception as exc:  # defensive: handlers should return a ToolResult
                result = ToolResult.error(
                    "TOOL_ERROR",
                    "工具执行失败，请稍后再试",
                    data={"error": str(exc)[:300]},
                    retryable=spec.retryable,
                )

            if result.ok or result.code not in RETRYABLE_CODES or attempt == attempts - 1:
                break
            # Only idempotent read tools are retried; write intents are never
            # retried because their handlers create pending actions.
            time.sleep(0.15)

        return self._finish(result or ToolResult.error("TOOL_ERROR", "工具执行失败"), started)

    @staticmethod
    def _finish(result: ToolResult, started: float) -> ToolResult:
        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result

    @staticmethod
    def masked_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        """Best-effort audit masking for fields we never want in logs."""
        blocked = ("phone", "address", "id_card", "password", "token")
        return {
            key: ("***" if any(word in key.lower() for word in blocked) else value)
            for key, value in (arguments or {}).items()
        }
