"""Verify the configured LLM endpoint supports OpenAI-style tool calling.

Usage:
    python scripts/smoke_llm_tools.py

Uses D_API_KEY / D_BASE_URL / D_MODEL from the environment (or .env loaded by
app.config).  No Flask app or database is needed.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Config  # noqa: E402

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    }
]


def call(messages: list[dict]) -> dict:
    payload = json.dumps(
        {
            "model": Config.LLM_MODEL,
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{Config.LLM_BASE_URL.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {Config.LLM_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    if not Config.LLM_API_KEY:
        raise SystemExit("D_API_KEY is not configured")

    print(f"endpoint: {Config.LLM_BASE_URL}")
    print(f"model   : {Config.LLM_MODEL}")

    messages = [
        {"role": "user", "content": "What is the weather in Beijing? Use the tool."}
    ]
    first = call(messages)
    choice = first["choices"][0]
    message = choice["message"]
    print(f"finish_reason: {choice.get('finish_reason')}")
    print(f"tool_calls  : {json.dumps(message.get('tool_calls'), ensure_ascii=False)}")

    if not message.get("tool_calls"):
        raise SystemExit("FAILED: the endpoint did not return tool_calls")

    messages.append(
        {
            "role": "assistant",
            "content": message.get("content") or "",
            "tool_calls": message["tool_calls"],
        }
    )
    for tool_call in message["tool_calls"]:
        arguments = json.loads(tool_call["function"]["arguments"])
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(
                    {"city": arguments.get("city"), "weather": "sunny", "temp_c": 22}
                ),
            }
        )

    second = call(messages)
    print(f"final answer: {second['choices'][0]['message'].get('content')}")
    print("OK: tool calling is supported")


if __name__ == "__main__":
    main()
