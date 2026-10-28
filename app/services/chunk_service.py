"""
Chunk service for chunk detail and editing.
"""
from __future__ import annotations

from typing import Optional

from app.config import Config
from app.extensions import db
from app.models.chunk import Chunk
from app.services.model_cache import get_milvus_client
from app.services.ollama_service import OllamaService


class ChunkService:
    """Chunk detail and edit operations."""

    @staticmethod
    def get_chunk(chunk_id: int) -> Optional[Chunk]:
        """Get chunk by id."""
        return db.session.get(Chunk, chunk_id)

    @staticmethod
    def update_chunk(
        chunk_id: int,
        content: str,
        tags_text: str = "",
        note: str = "",
    ) -> tuple[bool, str, Optional[Chunk]]:
        """Update chunk content and metadata, and sync vector if content changes."""
        chunk = db.session.get(Chunk, chunk_id)
        if not chunk:
            return False, "Chunk 不存在", None

        content = content.strip()
        if not content:
            return False, "Chunk 内容不能为空", None

        metadata = chunk.metadata_dict
        metadata["tags"] = ChunkService._parse_tags(tags_text)
        metadata["note"] = note.strip()

        content_changed = chunk.content != content
        chunk.content = content
        chunk.chunk_metadata = metadata

        if content_changed:
            ChunkService._sync_chunk_vector(chunk)

        db.session.commit()
        return True, "Chunk 更新成功", chunk

    @staticmethod
    def _parse_tags(tags_text: str) -> list[str]:
        """Parse comma or newline separated tags."""
        raw_tags = tags_text.replace("\n", ",").split(",")
        result: list[str] = []
        seen = set()
        for tag in raw_tags:
            normalized = tag.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    @staticmethod
    def _sync_chunk_vector(chunk: Chunk):
        """Re-embed a single chunk and replace its Milvus record."""
        embedding_model = (
            (chunk.knowledge_base.default_embedding_model if chunk.knowledge_base else None)
            or Config.OLLAMA_EMBEDDING_MODEL
        )
        vector = OllamaService.embed_query(chunk.content, model=embedding_model)

        client = get_milvus_client(Config.MILVUS_URI, Config.MILVUS_TOKEN)

        if chunk.vector_id:
            try:
                client.delete(
                    collection_name=Config.MILVUS_COLLECTION,
                    filter=f"id == {int(chunk.vector_id)}",
                )
            except Exception:
                pass

        result = client.insert(
            collection_name=Config.MILVUS_COLLECTION,
            data=[
                {
                    "knowledge_base_id": chunk.knowledge_base_id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.content,
                    "dense": vector,
                }
            ],
        )
        client.flush(collection_name=Config.MILVUS_COLLECTION)
        client.load_collection(collection_name=Config.MILVUS_COLLECTION)

        if isinstance(result, dict) and "ids" in result and result["ids"]:
            chunk.vector_id = str(result["ids"][0])
        elif result:
            chunk.vector_id = str(result[0]["id"])
