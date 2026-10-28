"""Bounded agent loop.

The loop is deliberately explicit and synchronous: no LangGraph, no hidden
state machine.  It stops at ``AGENT_MAX_STEPS`` tool executions, and it stops
immediately when a write intent creates a PendingAction.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from app.agents.context import AgentContext
from app.agents.schemas import AgentState, TurnAccumulator
from app.config import Config
from app.tools.base import ToolResult
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


class AgentRuntime:
    def __init__(
        self,
        gateway,
        executor: ToolExecutor,
        registry: ToolRegistry,
        max_steps: int | None = None,
    ) -> None:
        self.gateway = gateway
        self.executor = executor
        self.registry = registry
        self.max_steps = max_steps or Config.AGENT_MAX_STEPS

    def run(self, state: AgentState, context: AgentContext) -> Iterator[dict[str, Any]]:
        """Run until a final answer, a confirmation pause, or an error."""
        while state.step_count < self.max_steps:
            accumulator = TurnAccumulator()
            for event in self.gateway.stream_turn(
                state.messages,
                tools=self.registry.planning_schemas(),
            ):
                event_type = event.get("type")
                if event_type == "error":
                    yield {"type": "error", "message": event.get("message") or "模型调用失败"}
                    return
                accumulator.feed(event)
                if event_type == "content" and event.get("content"):
                    # Streamed immediately; the frontend clears this bubble if the
                    # turn ends with a tool call (see discard_content below).
                    yield {"type": "delta", "content": event["content"]}
                elif event_type == "reasoning" and event.get("content"):
                    yield {"type": "reasoning", "content": event["content"]}

            turn = accumulator.finalize()
            if not turn.tool_calls:
                answer = turn.content.strip()
                if not answer:
                    yield {"type": "error", "message": "模型未返回有效回答，请重试"}
                    return
                state.messages.append({"role": "assistant", "content": answer})
                yield {"type": "end", "content": answer}
                return

            # The assistant message with tool_calls must be replayed verbatim.
            state.messages.append(turn.assistant_message())
            has_preamble = accumulator.has_content

            for position, call in enumerate(turn.tool_calls):
                if state.step_count >= self.max_steps:
                    # Keep the message sequence valid: every tool_call_id needs
                    # a tool message before the run stops.
                    for skipped in turn.tool_calls[position:]:
                        skipped_result = ToolResult.error(
                            "MAX_STEPS_REACHED",
                            "已达到本次处理的最大工具调用步数，请转人工处理",
                        )
                        state.messages.append(self._tool_message(skipped.id, skipped_result))
                    yield {
                        "type": "error",
                        "message": "已达到最大工具调用步数，请转人工处理",
                    }
                    return

                state.step_count += 1
                spec = self.registry.get(call.name)
                yield {
                    "type": "tool_start",
                    "name": call.name,
                    "tool_call_id": call.id,
                    "label": spec.label if spec else call.name,
                    "discard_content": bool(has_preamble and position == 0),
                }

                result = self.executor.execute(call, context, state)
                # The pending call also gets a placeholder tool message so the
                # replayed conversation keeps tool_call ids in order.  On
                # confirm, AgentService replaces this message in place.
                state.messages.append(self._tool_message(call.id, result))
                yield {
                    "type": "tool_result",
                    "name": call.name,
                    "tool_call_id": call.id,
                    "step_index": state.step_count,
                    "ok": result.ok,
                    "code": result.code,
                    "display_text": result.display_text,
                    "sources": result.sources,
                    "duration_ms": result.duration_ms,
                    "arguments": ToolExecutor.masked_arguments(call.arguments),
                }

                if result.pending_action_id:
                    state.pending_action_id = result.pending_action_id
                    # Any remaining calls in this turn are skipped, but they
                    # still need tool messages for a valid conversation.
                    for skipped in turn.tool_calls[position + 1:]:
                        skipped_result = ToolResult.error(
                            "SKIPPED_PENDING_CONFIRMATION",
                            "因等待用户确认，本次未执行该工具",
                        )
                        state.messages.append(self._tool_message(skipped.id, skipped_result))
                    yield {
                        "type": "confirmation_required",
                        "action": result.pending_action or {},
                        "action_id": result.pending_action_id,
                    }
                    return

        yield {"type": "error", "message": "已达到最大工具调用步数，请转人工处理"}

    @staticmethod
    def _tool_message(tool_call_id: str, result: ToolResult) -> dict[str, Any]:
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result.to_message(),
        }
