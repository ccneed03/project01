"""
Document Service
"""
import os
import re
import json
from typing import Optional, List, Tuple
from datetime import datetime
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader
from pymilvus import MilvusClient, DataType, Function, FunctionType
from app.config import Config
from app.models.document import Document
from app.models.chunk import Chunk
from app.models.knowledge_base import KnowledgeBase
from app.models.document_index_task import DocumentIndexTask
from app.models.assistant import Assistant, AssistantKnowledgeBaseConfig
from app.services.model_cache import get_milvus_client
from app.services.ollama_service import OllamaService
from app.extensions import db


class DocumentService:
    """Document service for processing documents"""

    ALLOWED_EXTENSIONS = {"txt", "pdf", "docx", "md"}
    ACTIVE_STATUSES = {"previewing", "indexing", "pending", "parsing", "splitting", "vectorizing"}

    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        return "." in filename and filename.rsplit(".", 1)[1].lower() in DocumentService.ALLOWED_EXTENSIONS

    @staticmethod
    def create_document(
        knowledge_base_id: int,
        file: FileStorage,
        created_by: int,
    ) -> Tuple[bool, str, Optional[Document]]:
        """Create a new document record

        Returns:
            (success, message, document)
        """
        kb = db.session.get(KnowledgeBase, knowledge_base_id)
        if not kb:
            return False, "知识库不存在", None

        if not DocumentService.allowed_file(file.filename):
            return False, "不支持的文件类型", None

        # Generate secure filename
        original_filename = file.filename
        ext = original_filename.rsplit(".", 1)[1].lower()
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{secure_filename(original_filename)}"

        # Create directory
        upload_dir = os.path.join(Config.UPLOAD_FOLDER, "knowledge_bases", str(knowledge_base_id), "documents")
        os.makedirs(upload_dir, exist_ok=True)

        # Save file
        file_path = os.path.join(upload_dir, filename)
        file.save(file_path)

        # Get file size
        file_size = os.path.getsize(file_path)

        # Create document record
        doc = Document(
            knowledge_base_id=knowledge_base_id,
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_type=ext,
            file_size=file_size,
            status="uploaded",
            created_by=created_by,
        )

        db.session.add(doc)
        db.session.commit()

        return True, "文档上传成功", doc

    @staticmethod
    def _extract_text(doc: Document) -> str:
        """Extract text from document based on file type"""
        file_path = doc.file_path
        file_type = doc.file_type

        if file_type == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()
            return "\n\n".join([d.page_content for d in docs])

        elif file_type == "pdf":
            loader = PyMuPDFLoader(file_path)
            docs = loader.load()
            return "\n\n".join([d.page_content for d in docs])

        elif file_type in ["docx", "doc"]:
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            return "\n\n".join([d.page_content for d in docs])

        elif file_type == "md":
            loader = UnstructuredMarkdownLoader(file_path)
            docs = loader.load()
            return "\n\n".join([d.page_content for d in docs])

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    @staticmethod
    def _split_text(text: str, chunk_size: int, chunk_overlap: int, split_separator: str = "\n\n", chunk_mode: str = "paragraph") -> List[str]:
        """Split text into chunks"""
        if chunk_mode == "separator":
            # Separator mode: split purely by custom separators via regex, no size constraint
            pattern = DocumentService._build_separator_pattern(split_separator)
            chunks = re.split(pattern, text)
            # Filter out empty/whitespace-only chunks
            return [c for c in chunks if c.strip()]
        else:
            # Paragraph mode: size-based recursive splitting with separator cascade
            separators = DocumentService._build_separators(split_separator)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                separators=separators,
            )
            return splitter.split_text(text)

    @staticmethod
    def _build_separator_pattern(split_separator: str | None) -> str:
        """Build a regex OR pattern from | delimited separator input.

        The user can use | to specify multiple separators, e.g. "。|." means
        split at both Chinese and English period. Each separator is regex-escaped
        and combined into a single regex OR pattern for re.split().
        """
        if not split_separator:
            return r"\n\n|\n"

        # Split by | to get individual separators
        raw_separators = split_separator.split("|")
        escaped: list[str] = []
        for sep in raw_separators:
            sep = sep.strip()
            if not sep:
                continue
            # Normalize escaped form input (e.g. \n\n → real newline)
            normalized = DocumentService._normalize_separator(sep)
            # Escape for regex so special chars like . are literal
            escaped.append(re.escape(normalized))

        if not escaped:
            return r"\n\n|\n"

        return "|".join(escaped)

    @staticmethod
    def _build_separators(split_separator: str | None) -> List[str]:
        """Build a robust separator list from configured text input."""
        default_separators = ["\n\n", "\n", "。", "！", "？", "；", ";", "，", ",", " ", ""]
        separators: list[str] = []

        if split_separator:
            normalized = DocumentService._normalize_separator(split_separator)
            if normalized:
                separators.append(normalized)

        for separator in default_separators:
            if separator not in separators:
                separators.append(separator)

        return separators

    @staticmethod
    def _normalize_separator(split_separator: str) -> str:
        """Convert escaped form input like '\\n\\n' into real separators."""
        try:
            return bytes(split_separator, "utf-8").decode("unicode_escape")
        except UnicodeDecodeError:
            return split_separator

    @staticmethod
    def _store_to_milvus(
        kb: KnowledgeBase,
        doc_id: int,
        chunks: List[str],
        embedding_model: str | None = None,
    ) -> List[int]:
        """Store chunks to Milvus

        Returns:
            List of vector IDs
        """
        # Generate vectors through the locally deployed Ollama service.
        vectors = OllamaService.embed_documents(
            chunks,
            model=embedding_model or kb.default_embedding_model or Config.OLLAMA_EMBEDDING_MODEL,
        )

        # Connect to Milvus
        client = get_milvus_client(Config.MILVUS_URI, Config.MILVUS_TOKEN)

        DocumentService._ensure_collection(client)

        # Prepare data
        data = [
            {
                "knowledge_base_id": kb.id,
                "document_id": doc_id,
                "chunk_index": index,
                "text": chunk,
                "dense": vector,
            }
            for index, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]

        # Insert
        result = client.insert(
            collection_name=Config.MILVUS_COLLECTION,
            data=data,
        )
        client.flush(collection_name=Config.MILVUS_COLLECTION)
        client.load_collection(collection_name=Config.MILVUS_COLLECTION)

        if isinstance(result, dict) and "ids" in result:
            return [int(vector_id) for vector_id in result["ids"]]

        return [r["id"] for r in result]

    @staticmethod
    def _ensure_collection(client: MilvusClient):
        """Ensure the Milvus collection exists with the required schema."""
        if client.has_collection(Config.MILVUS_COLLECTION):
            description = client.describe_collection(Config.MILVUS_COLLECTION)
            field_names = {field["name"] for field in description.get("fields", [])}
            required = {"knowledge_base_id", "document_id", "chunk_index", "text", "dense", "sparse"}
            if not required.issubset(field_names):
                raise ValueError(
                    f"Milvus collection '{Config.MILVUS_COLLECTION}' 已存在，但 schema 与 app 不兼容。"
                    "请更换 collection 名称或删除旧 collection 后重建。"
                )
            dense_field = next(field for field in description.get("fields", []) if field["name"] == "dense")
            current_dimension = int((dense_field.get("params") or {}).get("dim", 0))
            if current_dimension and current_dimension != Config.EMBEDDING_DIM:
                raise ValueError(
                    f"Milvus collection '{Config.MILVUS_COLLECTION}' 的 dense 向量维度为 {current_dimension}，"
                    f"与 Ollama Embedding 配置的 {Config.EMBEDDING_DIM} 不一致。"
                    "请删除旧 collection 后重建，并重新索引文档。"
                )
            return

        schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="knowledge_base_id", datatype=DataType.INT64)
        schema.add_field(field_name="document_id", datatype=DataType.INT64)
        schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
        schema.add_field(
            field_name="text",
            datatype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            analyzer_params={"type": "chinese"},
        )
        schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=Config.EMBEDDING_DIM)
        schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

        schema.add_function(
            Function(
                name="text_bm25_emb",
                input_field_names=["text"],
                output_field_names=["sparse"],
                function_type=FunctionType.BM25,
            )
        )

        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense", index_type="AUTOINDEX", metric_type="L2")
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

        client.create_collection(
            collection_name=Config.MILVUS_COLLECTION,
            schema=schema,
            index_params=index_params,
        )
        client.load_collection(collection_name=Config.MILVUS_COLLECTION)

    @staticmethod
    def _save_chunks(
        doc: Document,
        kb_id: int,
        chunks: List[str],
        vector_ids: List[int],
        index_task_id: int | None = None,
    ):
        """Save chunk metadata to database"""
        for i, (chunk_content, vector_id) in enumerate(zip(chunks, vector_ids)):
            chunk = Chunk(
                document_id=doc.id,
                knowledge_base_id=kb_id,
                index_task_id=index_task_id,
                content=chunk_content,
                vector_id=str(vector_id),
                chunk_index=i,
                chunk_metadata={"original_file": doc.original_filename},
            )
            db.session.add(chunk)

        db.session.commit()

    @staticmethod
    def delete_document(doc_id: int) -> Tuple[bool, str]:
        """Delete document and related data

        Returns:
            (success, message)
        """
        doc = db.session.get(Document, doc_id)
        if not doc:
            return False, "文档不存在"

        bindings = (
            AssistantKnowledgeBaseConfig.query
            .join(Assistant, Assistant.id == AssistantKnowledgeBaseConfig.assistant_id)
            .filter(
                AssistantKnowledgeBaseConfig.knowledge_base_id == doc.knowledge_base_id,
                Assistant.is_deleted.is_(False),
            )
            .all()
        )
        if bindings:
            assistant_names = [binding.assistant.name for binding in bindings if binding.assistant]
            assistant_names = list(dict.fromkeys(assistant_names))
            preview_names = "、".join(assistant_names[:3])
            if len(assistant_names) > 3:
                preview_names += f" 等 {len(assistant_names)} 个智能助手"
            return False, f"当前知识库正在被智能助手使用：{preview_names}。请先删除智能助手后再删除文档。"

        try:
            # Delete from Milvus
            DocumentService._delete_from_milvus(doc)

            # Delete chunks
            Chunk.query.filter_by(document_id=doc.id).delete()

            # Delete file
            if doc.file_path and os.path.exists(doc.file_path):
                os.remove(doc.file_path)

            # Soft delete document
            doc.is_deleted = True
            db.session.commit()

            return True, "文档删除成功"

        except Exception as e:
            return False, f"文档删除失败: {str(e)}"

    @staticmethod
    def _delete_from_milvus(doc: Document):
        """Delete document chunks from Milvus"""
        kb = doc.knowledge_base
        if not kb:
            return

        try:
            client = get_milvus_client(Config.MILVUS_URI, Config.MILVUS_TOKEN)
            if client.has_collection(Config.MILVUS_COLLECTION):
                client.delete(
                    collection_name=Config.MILVUS_COLLECTION,
                    filter=f"knowledge_base_id == {kb.id} and document_id == {doc.id}",
                )
        except Exception:
            pass  # Ignore errors if Milvus delete is not supported

    @staticmethod
    def get_document(doc_id: int) -> Optional[Document]:
        """Get document by ID"""
        return Document.query.filter_by(id=doc_id, is_deleted=False).first()

    @staticmethod
    def list_documents(
        knowledge_base_id: int = None,
        page: int = 1,
        per_page: int = 20,
        status: str = None,
        search: str = None,
    ) -> Tuple[List[Document], int]:
        """List documents with filters

        Returns:
            (documents, total_count)
        """
        query = Document.query.filter_by(is_deleted=False)

        if knowledge_base_id:
            query = query.filter_by(knowledge_base_id=knowledge_base_id)
        if status == "processing":
            query = query.filter(Document.status.in_(sorted(DocumentService.ACTIVE_STATUSES - {"pending"})))
        elif status:
            query = query.filter_by(status=status)
        if search:
            query = query.filter(Document.original_filename.contains(search))

        total = query.count()
        docs = query.order_by(Document.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        ).items

        return docs, total

    @staticmethod
    def serialize_document(doc: Document) -> dict:
        """Serialize document for polling API responses."""
        preview_task = None
        current_task = None
        if getattr(doc, "index_tasks", None):
            preview_task = next((task for task in doc.index_tasks if task.task_type == "preview"), None)
            current_task = next((task for task in doc.index_tasks if task.task_type == "index"), None)

        return {
            "id": doc.id,
            "title": doc.display_title,
            "original_filename": doc.original_filename,
            "file_type": doc.file_type.upper(),
            "file_size_formatted": doc.file_size_formatted,
            "status": doc.status,
            "status_label": doc.status_label,
            "chunk_count": doc.chunk_count or 0,
            "current_config_id": doc.current_config_id,
            "current_index_task_id": doc.current_index_task_id,
            "preview_task_id": preview_task.id if preview_task else None,
            "preview_task_status": preview_task.status if preview_task else None,
            "preview_task_progress": preview_task.progress if preview_task else None,
            "index_task_id": current_task.id if current_task else doc.current_index_task_id,
            "index_task_status": current_task.status if current_task else None,
            "index_task_progress": current_task.progress if current_task else None,
            "created_at": doc.created_at.strftime("%Y-%m-%d %H:%M:%S") if doc.created_at else "",
            "view_url": f"/documents/{doc.id}",
            "preview_url": f"/documents/{doc.id}/preview",
            "chunks_url": f"/documents/{doc.id}/chunks",
            "active": doc.status in DocumentService.ACTIVE_STATUSES,
        }

    @staticmethod
    def serialize_index_task(task: DocumentIndexTask) -> dict:
        """Serialize index task for APIs."""
        result_summary = task.result_summary
        if isinstance(result_summary, str):
            try:
                result_summary = json.loads(result_summary)
            except json.JSONDecodeError:
                result_summary = {"raw": result_summary}

        return {
            "id": task.id,
            "document_id": task.document_id,
            "knowledge_base_id": task.knowledge_base_id,
            "config_id": task.config_id,
            "task_type": task.task_type,
            "status": task.status,
            "progress": task.progress,
            "error_message": task.error_message,
            "result_summary": result_summary,
            "started_at": task.started_at.strftime("%Y-%m-%d %H:%M:%S") if task.started_at else None,
            "finished_at": task.finished_at.strftime("%Y-%m-%d %H:%M:%S") if task.finished_at else None,
            "created_at": task.created_at.strftime("%Y-%m-%d %H:%M:%S") if task.created_at else None,
        }

    @staticmethod
    def get_document_chunks(
        doc_id: int,
        page: int = 1,
        per_page: int = 50,
    ) -> Tuple[List[Chunk], int]:
        """Get chunks for a document

        Returns:
            (chunks, total_count)
        """
        query = Chunk.query.filter_by(document_id=doc_id)
        total = query.count()
        chunks = query.order_by(Chunk.chunk_index).paginate(
            page=page, per_page=per_page, error_out=False
        ).items

        return chunks, total
