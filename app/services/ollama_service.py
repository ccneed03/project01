"""Ollama API integration for embedding generation."""
from __future__ import annotations

from typing import Iterable

from app.config import Config
from app.services.model_cache import get_ollama_http_client


class OllamaService:
    """Access locally deployed Ollama models through its HTTP API."""

    @staticmethod
    def list_embedding_models() -> list[str]:
        """Return installed Ollama models advertised for embedding use."""
        try:
            response = get_ollama_http_client(Config.OLLAMA_BASE_URL).get("/api/tags")
            response.raise_for_status()
        except Exception:
            return []

        models: list[str] = []
        for item in response.json().get("models", []):
            name = item.get("name")
            capabilities = item.get("capabilities") or []
            if not name or "embedding" not in capabilities:
                continue
            # Rerank models can expose an embedding capability, but must not be
            # selectable as document embedding models.
            if "rerank" in name.lower():
                continue
            models.append(name)
        return sorted(models)

    @staticmethod
    def embedding_model_exists(model: str) -> bool:
        """Check that an exact Ollama model name is locally installed."""
        return model in OllamaService.list_embedding_models()

    @staticmethod
    def embed_documents(texts: Iterable[str], model: str | None = None) -> list[list[float]]:
        """Generate document embeddings with Ollama's ``/api/embed`` endpoint."""
        inputs = list(texts)
        if not inputs:
            return []

        model = (model or Config.OLLAMA_EMBEDDING_MODEL).strip()
        if not model:
            raise ValueError("未配置 Ollama Embedding 模型")

        response = get_ollama_http_client(Config.OLLAMA_BASE_URL).post(
            "/api/embed",
            json={"model": model, "input": inputs},
        )
        response.raise_for_status()
        embeddings = response.json().get("embeddings")

        if not isinstance(embeddings, list) or len(embeddings) != len(inputs):
            raise ValueError("Ollama Embedding 响应异常")
        OllamaService._validate_dimensions(embeddings, model)
        return embeddings

    @staticmethod
    def embed_query(query: str, model: str | None = None) -> list[float]:
        """Generate one query embedding."""
        return OllamaService.embed_documents([query], model=model)[0]

    @staticmethod
    def _validate_dimensions(embeddings: list[list[float]], model: str):
        """Ensure Ollama vectors match the configured Milvus schema dimension."""
        dimensions = {len(vector) for vector in embeddings if isinstance(vector, list)}
        if len(dimensions) != 1 or not dimensions or 0 in dimensions:
            raise ValueError("Ollama Embedding 响应包含无效向量")

        actual_dimension = dimensions.pop()
        if actual_dimension != Config.EMBEDDING_DIM:
            raise ValueError(
                f"Ollama 模型 '{model}' 返回 {actual_dimension} 维向量，"
                f"但当前 EMBEDDING_DIM 配置为 {Config.EMBEDDING_DIM}。"
                "请修改配置并重建 Milvus collection 后重新索引文档。"
            )
