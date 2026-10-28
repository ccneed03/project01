"""
Model catalog service for local embedding and reranker directories.
"""
from pathlib import Path

from app.services.ollama_service import OllamaService


class ModelCatalogService:
    """Expose available local model directories for UI dropdowns."""

    RERANK_ROOT = Path("/Users/jiang/models/rerank_models")

    @staticmethod
    def list_embedding_models() -> list[str]:
        """Return locally installed Ollama embedding model names."""
        return OllamaService.list_embedding_models()

    @staticmethod
    def list_rerank_models() -> list[str]:
        """Return available reranker model directory names."""
        return ModelCatalogService._list_model_dirs(ModelCatalogService.RERANK_ROOT)

    @staticmethod
    def resolve_embedding_model(model_name: str | None) -> str | None:
        """Validate an embedding model name against the local Ollama registry."""
        normalized = (model_name or "").strip()
        return normalized if normalized and OllamaService.embedding_model_exists(normalized) else None

    @staticmethod
    def resolve_rerank_model(model_name: str | None) -> str | None:
        """Resolve a rerank model folder name to an absolute path."""
        if not model_name:
            return None
        path = ModelCatalogService.RERANK_ROOT / model_name
        return str(path) if path.exists() and path.is_dir() else None

    @staticmethod
    def _list_model_dirs(root: Path) -> list[str]:
        """List first-level child directories, ignoring hidden entries."""
        if not root.exists() or not root.is_dir():
            return []

        names = [
            child.name
            for child in root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ]
        return sorted(names)
