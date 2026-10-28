"""
Knowledge Base Service
"""
from typing import Optional, List
from pymilvus import MilvusClient, connections
from app.models.knowledge_base import KnowledgeBase
from app.extensions import db


class KnowledgeService:
    """Knowledge Base service"""

    @staticmethod
    def create_knowledge_base(
        name: str,
        description: str = None,
        created_by: int = None,
        default_embedding_model: str = None,
        split_separator: str = "\n\n",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        retrieval_method: str = "hybrid",
        rerank_k: int = 6,
        enable_rerank: bool = True,
    ) -> tuple[bool, str, Optional[KnowledgeBase]]:
        """Create a new knowledge base

        Returns:
            (success, message, knowledge_base)
        """
        kb = KnowledgeBase(
            name=name,
            description=description,
            created_by=created_by,
            default_embedding_model=default_embedding_model,
            split_separator=split_separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_method=retrieval_method,
            rerank_k=rerank_k,
            enable_rerank=enable_rerank,
        )

        db.session.add(kb)
        db.session.commit()

        return True, "知识库创建成功", kb

    @staticmethod
    def update_knowledge_base(
        kb_id: int,
        name: str = None,
        description: str = None,
        default_embedding_model: str = None,
        split_separator: str = None,
        chunk_size: int = None,
        chunk_overlap: int = None,
        retrieval_method: str = None,
        rerank_k: int = None,
        enable_rerank: bool = None,
        status: str = None,
    ) -> tuple[bool, str, Optional[KnowledgeBase]]:
        """Update a knowledge base

        Returns:
            (success, message, knowledge_base)
        """
        kb = db.session.get(KnowledgeBase, kb_id)
        if not kb:
            return False, "知识库不存在", None

        if name:
            kb.name = name
        if description is not None:
            kb.description = description
        if default_embedding_model is not None:
            kb.default_embedding_model = default_embedding_model
        if split_separator is not None:
            kb.split_separator = split_separator
        if chunk_size is not None:
            kb.chunk_size = chunk_size
        if chunk_overlap is not None:
            kb.chunk_overlap = chunk_overlap
        if retrieval_method is not None:
            kb.retrieval_method = retrieval_method
        if rerank_k is not None:
            kb.rerank_k = rerank_k
        if enable_rerank is not None:
            kb.enable_rerank = enable_rerank
        if status is not None:
            kb.status = status

        db.session.commit()

        return True, "知识库更新成功", kb

    @staticmethod
    def delete_knowledge_base(kb_id: int) -> tuple[bool, str]:
        """Soft delete a knowledge base

        Returns:
            (success, message)
        """
        kb = db.session.get(KnowledgeBase, kb_id)
        if not kb:
            return False, "知识库不存在"

        kb.is_deleted = True
        db.session.commit()

        return True, "知识库已删除"

    @staticmethod
    def get_knowledge_base(kb_id: int) -> Optional[KnowledgeBase]:
        """Get knowledge base by ID"""
        return KnowledgeBase.query.filter_by(id=kb_id, is_deleted=False).first()

    @staticmethod
    def list_knowledge_bases(
        page: int = 1,
        per_page: int = 20,
        search: str = None,
    ) -> tuple[List[KnowledgeBase], int]:
        """List knowledge bases with pagination

        Returns:
            (knowledge_bases, total_count)
        """
        query = KnowledgeBase.query.filter_by(is_deleted=False)

        if search:
            query = query.filter(KnowledgeBase.name.contains(search))

        total = query.count()
        kbs = query.order_by(KnowledgeBase.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        ).items

        return kbs, total

    @staticmethod
    def test_milvus_connection(milvus_uri: str, milvus_token: str) -> tuple[bool, str]:
        """Test Milvus connection

        Returns:
            (success, message)
        """
        from app.config import Config

        try:
            uri = milvus_uri or Config.MILVUS_URI
            token = milvus_token or Config.MILVUS_TOKEN
            connections.connect(uri=uri, token=token)
            connections.disconnect(alias="default")
            return True, "Milvus 连接成功"
        except Exception as e:
            return False, f"Milvus 连接失败: {str(e)}"
