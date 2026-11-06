"""Run one real, read-only Agent turn against the configured LLM."""
from __future__ import annotations

import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app import create_app  # noqa: E402
from app.agents.context import AgentContext  # noqa: E402
from app.agents.gateway import ChatGateway  # noqa: E402
from app.agents.prompts import build_system_prompt  # noqa: E402
from app.agents.runtime import AgentRuntime  # noqa: E402
from app.agents.schemas import AgentState  # noqa: E402
from app.models.assistant import Assistant  # noqa: E402
from app.models.user import User  # noqa: E402
from app.tools.executor import ToolExecutor  # noqa: E402
from app.tools.registry import build_default_registry  # noqa: E402


def main() -> int:
    app = create_app()
    with app.app_context():
        assistant = Assistant.query.filter_by(
            name="售后助手", is_deleted=False
        ).first()
        user = User.query.filter_by(username="admin").first()
        if not assistant or not user:
            print("[FAIL] demo user or assistant missing")
            return 1

        registry = build_default_registry()
        gateway = ChatGateway(model=assistant.llm_model)
        runtime = AgentRuntime(
            gateway=gateway,
            executor=ToolExecutor(registry),
            registry=registry,
        )
        context = AgentContext(
            user_id=user.id,
            session_id=0,
            trace_id=uuid.uuid4().hex,
            run_id=uuid.uuid4().hex,
            assistant_id=assistant.id,
            knowledge_base_id=assistant.knowledge_base.id,
            permissions={"after_sales.read", "after_sales.write"},
            retrieval_config={"retrieval_mode": "hybrid", "enable_rerank": False},
        )
        state = AgentState(
            messages=[
                {"role": "system", "content": build_system_prompt(assistant)},
                {
                    "role": "user",
                    "content": "请介绍 LAPTOP-001 的配置和保养方式，不需要办理退货。",
                },
            ]
        )

        tool_names: list[str] = []
        answer = ""
        error = ""
        for event in runtime.run(state, context):
            if event.get("type") == "tool_start":
                tool_names.append(event.get("name") or "unknown")
            elif event.get("type") == "end":
                answer = (event.get("content") or "").strip()
            elif event.get("type") == "error":
                error = event.get("message") or "unknown error"

        print(f"tools: {tool_names}")
        print(f"answer: {answer[:500]}")
        if error:
            print(f"[FAIL] {error}")
            return 1
        if "search_product" not in tool_names:
            print("[FAIL] model did not call search_product")
            return 1
        if not answer:
            print("[FAIL] empty final answer")
            return 1
        print("[ OK ] real Agent read-only turn completed")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
