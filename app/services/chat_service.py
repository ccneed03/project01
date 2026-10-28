"""
Chat Service
"""
import json
from collections.abc import Iterator
from typing import Any
from app.config import Config
from app.services.model_cache import get_http_client


class ChatService:
    """Chat service for LLM integration"""

    @staticmethod
    def generate_response(
        query: str,
        context: str,
        knowledge_base,
        assistant=None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> str:
        """Generate LLM response with RAG context

        Args:
            query: User query
            context: Retrieved context
            knowledge_base: KnowledgeBase instance
            model: Model name (default from config)
            temperature: Sampling temperature
            max_tokens: Max response tokens

        Returns:
            Generated response
        """
        api_key = Config.LLM_API_KEY
        if not api_key:
            return "LLM 未配置：请先在环境变量中设置 D_API_KEY。"

        try:
            return "".join(
                ChatService.stream_response(
                    query=query,
                    context=context,
                    knowledge_base=knowledge_base,
                    assistant=assistant,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    enable_thinking=enable_thinking,
                )
            ).strip()
        except Exception as exc:
            return f"LLM 调用失败：{exc}"

    @staticmethod
    def stream_response(
        query: str,
        context: str,
        knowledge_base,
        assistant=None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> Iterator[str]:
        """Stream LLM response chunks."""
        for chunk in ChatService.stream_events(
                query=query,
                context=context,
                knowledge_base=knowledge_base,
                assistant=assistant,
                model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
        ):
            if chunk["type"] == "content":
                yield chunk["content"]

    @staticmethod
    def stream_events(
        query: str,
        context: str,
        knowledge_base,
        assistant=None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Stream structured events from the LLM."""
        api_key = Config.LLM_API_KEY
        if not api_key:
            yield {"type": "content", "content": "LLM 未配置：请先在环境变量中设置 D_API_KEY。"}
            return

        messages = ChatService._build_messages(query, context, knowledge_base, assistant=assistant)
        payload = {
            "model": model or getattr(assistant, "llm_model", None) or Config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "enable_thinking": enable_thinking,
        }

        try:
            client = get_http_client(Config.LLM_BASE_URL, api_key)
            with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue

                    if not line.startswith("data:"):
                        continue

                    data = line[5:].strip()
                    if not data:
                        continue

                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning_content") or ""
                    if reasoning and enable_thinking:
                        yield {"type": "reasoning", "content": reasoning}

                    text = delta.get("content") or ""
                    if text:
                        yield {"type": "content", "content": text}
        except Exception as exc:
            yield {"type": "content", "content": f"LLM 调用失败：{exc}"}

    @staticmethod
    def stream_chat(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
        tool_choice: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Stream one assistant turn with optional OpenAI-compatible tools.

        Yields normalised events:
          - reasoning        {"content": ...}
          - content          {"content": ...}
          - tool_call_delta  {"index", "id", "name", "arguments"}
          - finish           {"reason": "tool_calls" | "stop"}
          - error            {"message": ...}
        """
        api_key = Config.LLM_API_KEY
        if not api_key:
            yield {"type": "error", "message": "LLM 未配置：请先在环境变量中设置 D_API_KEY。"}
            return

        payload: dict[str, Any] = {
            "model": model or Config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
            "enable_thinking": enable_thinking,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"

        finish_sent = False
        try:
            client = get_http_client(Config.LLM_BASE_URL, api_key)
            with client.stream("POST", "/chat/completions", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data:
                        continue
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}

                    reasoning = delta.get("reasoning_content") or ""
                    if reasoning and enable_thinking:
                        yield {"type": "reasoning", "content": reasoning}

                    text = delta.get("content") or ""
                    if text:
                        yield {"type": "content", "content": text}

                    for tool_call in delta.get("tool_calls") or []:
                        function = tool_call.get("function") or {}
                        yield {
                            "type": "tool_call_delta",
                            "index": tool_call.get("index", 0),
                            "id": tool_call.get("id"),
                            "name": function.get("name"),
                            "arguments": function.get("arguments") or "",
                        }

                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        finish_sent = True
                        yield {"type": "finish", "reason": finish_reason}
        except Exception as exc:
            yield {"type": "error", "message": f"LLM 调用失败：{exc}"}
            return

        if not finish_sent:
            yield {"type": "finish", "reason": "stop"}

    @staticmethod
    def complete(
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ) -> dict[str, Any]:
        """Non-streaming completion used after a confirmed action."""
        api_key = Config.LLM_API_KEY
        if not api_key:
            raise RuntimeError("LLM 未配置：请先在环境变量中设置 D_API_KEY。")

        payload: dict[str, Any] = {
            "model": model or Config.LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "enable_thinking": enable_thinking,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            client = get_http_client(Config.LLM_BASE_URL, api_key)
            response = client.post("/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"LLM 调用失败：{exc}") from exc

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("LLM 未返回有效结果")
        choice = choices[0]
        message = choice.get("message") or {}

        tool_calls = []
        for item in message.get("tool_calls") or []:
            function = item.get("function") or {}
            raw = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw)
                parse_error = None
            except json.JSONDecodeError as exc:
                arguments = {}
                parse_error = str(exc)
            tool_calls.append(
                {
                    "id": item.get("id"),
                    "name": function.get("name") or "",
                    "arguments": arguments,
                    "raw_arguments": raw,
                    "parse_error": parse_error,
                }
            )

        return {
            "content": (message.get("content") or "").strip(),
            "reasoning": (message.get("reasoning_content") or "").strip(),
            "tool_calls": tool_calls,
            "finish_reason": choice.get("finish_reason") or "stop",
        }

    @staticmethod
    def _build_messages(query: str, context: str, knowledge_base, assistant=None) -> list[dict[str, str]]:
        """Build chat messages for plain AI or knowledge-grounded modes."""
        if knowledge_base is None:
            system_prompt = (
                getattr(assistant, "system_prompt", None)
                or (
                    "你是一个专业、友好的 AI 助手。"
                    "请直接回答用户问题，优先给出清晰、结构化、可执行的答案。"
                    "如果用户的问题含糊，可以做合理假设并明确说明。"
                )
            )
            user_prompt = f"""用户问题：{query}

请直接回答，语言自然、简洁，必要时给出要点。"""
        else:
            system_prompt = (
                getattr(assistant, "system_prompt", None)
                or (
                    f"你是知识库「{knowledge_base.name}」的问答助手。"
                    "请严格依据检索到的上下文回答。"
                    "如果上下文不足以支持结论，就明确说明“根据当前知识库内容，暂时无法确定”。"
                    "不要把外部常识混入成确定事实。"
                )
            )
            user_prompt = f"""上下文信息：
{context or "（未检索到相关上下文）"}

用户问题：{query}

请基于上下文给出准确、可信、简洁的回答。"""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
