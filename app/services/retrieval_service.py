"""
Retrieval Service - Core RAG Retrieval Logic
Based on knowledge/part4.py
"""
from typing import List, Dict, Optional, Any
from pymilvus import MilvusClient
from pymilvus.exceptions import MilvusException
from app.config import Config
from app.models.chunk import Chunk
from app.models.document import Document
from app.extensions import db
from app.services.model_cache import get_reranker, get_milvus_client
from app.services.model_catalog_service import ModelCatalogService
from app.services.ollama_service import OllamaService


class RetrievalService:
    """Retrieval service for RAG"""

    def __init__(self, knowledge_base):
        """Initialize retrieval service

        Args:
            knowledge_base: KnowledgeBase model instance
        """
        self.kb = knowledge_base
        self.milvus_uri = Config.MILVUS_URI
        self.milvus_token = Config.MILVUS_TOKEN
        self.embedding_model = (self.kb.default_embedding_model or Config.OLLAMA_EMBEDDING_MODEL).strip()
        self.rerank_model_path = Config.RERANK_MODEL_PATH
        self.collection_name = Config.MILVUS_COLLECTION
        self._field_names: Optional[set[str]] = None

    def _get_milvus_client(self) -> MilvusClient:
        """Get Milvus client"""
        return get_milvus_client(self.milvus_uri, self.milvus_token)

    def search(
        self,
        query: str,
        method: str = None,
        top_k: int = None,
        enable_rerank: bool = None,
        rerank_model: str | None = None,
        score_threshold: float | None = None,
        strict: bool = False,
    ) -> List[Dict[str, Any]]:
        """Search knowledge base

        Args:
            query: Search query
            method: Search method (vector, bm25, hybrid), default from KB
            top_k: Number of results to return, default from KB rerank_k
            enable_rerank: Whether to enable reranking, default from KB

        Returns:
            List of search results with document info
        """
        # Use KB settings as defaults
        method = method or self.kb.retrieval_method or "hybrid"
        method = self._normalize_method(method)
        top_k = top_k or self.kb.rerank_k or 6
        enable_rerank = enable_rerank if enable_rerank is not None else self.kb.enable_rerank
        if not self._collection_ready(strict=strict):
            if strict:
                raise RuntimeError(f"Milvus collection '{self.collection_name}' 不存在")
            return []

        if method == "vector":
            results = self._search_vector(query, top_k, strict=strict)
        elif method == "bm25":
            results = self._search_bm25(query, top_k, strict=strict)
        elif method == "hybrid":
            # Hybrid: BM25 top_k + vector top_k -> merge -> rerank -> return top_k
            bm25_results = self._search_bm25(query, top_k, strict=strict)
            vector_results = self._search_vector(query, top_k, strict=strict)

            if enable_rerank:
                # Merge results and rerank
                merged = self._merge_results(bm25_results, vector_results)
                results = self._rerank(query, merged, top_k, rerank_model=rerank_model)
            else:
                # Use RRF fusion without rerank
                results = self._rrf_fusion(bm25_results, vector_results, top_k)
        else:
            results = []

        results = self._apply_score_threshold(
            hits=results,
            method=method,
            enable_rerank=enable_rerank,
            score_threshold=score_threshold,
        )
        return self._enrich_with_references(results)

    @staticmethod
    def _normalize_method(method: str) -> str:
        """Map user-facing method names to internal retrieval modes."""
        normalized = (method or "hybrid").strip().lower()
        aliases = {
            "semantic": "vector",
            "keyword": "bm25",
        }
        return aliases.get(normalized, normalized)

    def _collection_ready(self, strict: bool = False) -> bool:
        """Check whether the configured Milvus collection exists."""
        try:
            client = self._get_milvus_client()
            return client.has_collection(self.collection_name)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Milvus 不可用：{exc}") from exc
            return False

    def _get_field_names(self) -> set[str]:
        """Read and cache the current Milvus collection fields."""
        if self._field_names is not None:
            return self._field_names

        client = self._get_milvus_client()
        description = client.describe_collection(self.collection_name)
        self._field_names = {field["name"] for field in description.get("fields", [])}
        return self._field_names

    def _supports_metadata_fields(self) -> bool:
        """Whether the collection includes app-level metadata fields."""
        fields = self._get_field_names()
        required = {"knowledge_base_id", "document_id", "chunk_index"}
        return required.issubset(fields)

    def _build_filter(self) -> Optional[str]:
        """Restrict retrieval results to the current knowledge base."""
        if not self._supports_metadata_fields():
            return None
        return f"knowledge_base_id == {self.kb.id}"

    def _search_bm25(self, query: str, limit: int, strict: bool = False) -> List[Dict[str, Any]]:
        """BM25 search (sparse vector)"""
        client = self._get_milvus_client()

        try:
            output_fields = ["text"]
            if self._supports_metadata_fields():
                output_fields.extend(["knowledge_base_id", "document_id", "chunk_index"])

            results = client.search(
                collection_name=self.collection_name,
                data=[query],
                anns_field="sparse",
                limit=limit,
                filter=self._build_filter() or "",
                output_fields=output_fields,
            )

            hits = []
            for hit in results[0]:
                hits.append({
                    "text": hit["entity"]["text"],
                    "score": hit["distance"],
                    "id": hit["id"],
                    "milvus_id": hit["id"],
                    "knowledge_base_id": hit["entity"].get("knowledge_base_id"),
                    "document_id": hit["entity"].get("document_id"),
                    "chunk_index": hit["entity"].get("chunk_index"),
                    "source": "bm25",
                })
            return hits
        except MilvusException as exc:
            if strict:
                raise RuntimeError(f"Milvus BM25 检索失败：{exc}") from exc
            return []

    def _search_vector(self, query: str, limit: int, strict: bool = False) -> List[Dict[str, Any]]:
        """Vector search (dense vector)"""
        client = self._get_milvus_client()
        query_vec = OllamaService.embed_query(query, model=self.embedding_model)

        try:
            output_fields = ["text"]
            if self._supports_metadata_fields():
                output_fields.extend(["knowledge_base_id", "document_id", "chunk_index"])

            results = client.search(
                collection_name=self.collection_name,
                data=[query_vec],
                anns_field="dense",
                limit=limit,
                search_params={"metric_type": "L2"},
                filter=self._build_filter() or "",
                output_fields=output_fields,
            )

            hits = []
            for hit in results[0]:
                hits.append({
                    "text": hit["entity"]["text"],
                    "score": hit["distance"],
                    "id": hit["id"],
                    "milvus_id": hit["id"],
                    "knowledge_base_id": hit["entity"].get("knowledge_base_id"),
                    "document_id": hit["entity"].get("document_id"),
                    "chunk_index": hit["entity"].get("chunk_index"),
                    "source": "vector",
                })
            return hits
        except MilvusException as exc:
            if strict:
                raise RuntimeError(f"Milvus 向量检索失败：{exc}") from exc
            return []

    def _merge_results(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
    ) -> List[Dict[str, Any]]:
        """Merge BM25 and vector results, deduplicate by milvus_id"""
        seen = set()
        merged = []

        for hit in bm25_results:
            if hit["milvus_id"] not in seen:
                seen.add(hit["milvus_id"])
                merged.append(hit)

        for hit in vector_results:
            if hit["milvus_id"] not in seen:
                seen.add(hit["milvus_id"])
                merged.append(hit)

        return merged

    def _rrf_fusion(
        self,
        bm25_results: List[Dict],
        vector_results: List[Dict],
        top_k: int,
        k: int = 60,
    ) -> List[Dict[str, Any]]:
        """RRF (Reciprocal Rank Fusion) fusion without rerank

        Args:
            bm25_results: BM25 search results
            vector_results: Vector search results
            top_k: Number of results to return
            k: RRF parameter (default 60)
        """
        rrf_scores = {}

        # BM25 scores
        for rank, hit in enumerate(bm25_results):
            milvus_id = hit["milvus_id"]
            rrf_scores[milvus_id] = rrf_scores.get(milvus_id, 0) + 1.0 / (k + rank + 1)
            if milvus_id not in rrf_scores or hit not in [h for h in rrf_scores if h == milvus_id]:
                pass

        # Vector scores
        for rank, hit in enumerate(vector_results):
            milvus_id = hit["milvus_id"]
            rrf_scores[milvus_id] = rrf_scores.get(milvus_id, 0) + 1.0 / (k + rank + 1)

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Build result list
        seen = set()
        results = []
        milvus_id_to_hit = {h["milvus_id"]: h for h in bm25_results + vector_results}

        for milvus_id, rrf_score in sorted_ids[:top_k * 2]:
            if milvus_id not in seen:
                seen.add(milvus_id)
                hit = milvus_id_to_hit.get(milvus_id)
                if hit:
                    hit["score"] = rrf_score
                    hit["rrf_score"] = rrf_score
                    results.append(hit)

        return results[:top_k]

    def _rerank(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: int,
        rerank_model: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Rerank results using BGE-reranker

        Unlike RRF fusion, rerank takes both BM25 and vector results as-is
        and reorders them based on semantic relevance.
        """
        if not hits:
            return []

        import torch

        rerank_path = ModelCatalogService.resolve_rerank_model(rerank_model) or self.rerank_model_path
        tokenizer, model = get_reranker(rerank_path)

        texts = [h["text"] for h in hits]
        pairs = [[query, text] for text in texts]

        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512)
            scores = model(**inputs, return_dict=True).logits.view(-1,).float()

        scores = scores.tolist()

        scored = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)

        result = []
        for hit, score in scored[:top_k]:
            hit["score"] = score
            result.append(hit)

        return result

    @staticmethod
    def _apply_score_threshold(
        hits: List[Dict[str, Any]],
        method: str,
        enable_rerank: bool,
        score_threshold: float | None,
    ) -> List[Dict[str, Any]]:
        """Filter hits using a mode-aware score threshold."""
        if score_threshold is None:
            return hits

        if method == "vector" and not enable_rerank:
            return [hit for hit in hits if float(hit.get("score", 0)) <= score_threshold]

        return [hit for hit in hits if float(hit.get("score", 0)) >= score_threshold]

    def _enrich_with_references(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich hits with document and chunk information"""
        for hit in hits:
            if hit.get("document_id") is not None and hit.get("chunk_index") is not None:
                chunk = Chunk.query.filter_by(
                    knowledge_base_id=self.kb.id,
                    document_id=hit.get("document_id"),
                    chunk_index=hit.get("chunk_index"),
                ).first()
            else:
                chunk = Chunk.query.filter_by(
                    knowledge_base_id=self.kb.id,
                    content=hit["text"],
                ).first()

            if chunk:
                document = db.session.get(Document, chunk.document_id)
                hit["chunk_id"] = chunk.id
                hit["chunk_index"] = chunk.chunk_index
                hit["document"] = {
                    "id": document.id if document else None,
                    "filename": document.filename if document else None,
                    "original_filename": document.original_filename if document else None,
                    "file_type": document.file_type if document else None,
                }
            else:
                hit["chunk_id"] = None
                hit["chunk_index"] = None
                hit["document"] = None

        return hits
