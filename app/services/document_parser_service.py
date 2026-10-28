"""
Document parsing and chunk preview service.
"""
from app.models.document import Document
from app.services.document_service import DocumentService


class DocumentParserService:
    """Reusable parser/splitter for preview and indexing flows."""

    @staticmethod
    def extract_text(document: Document) -> str:
        """Extract plain text from a stored document."""
        return DocumentService._extract_text(document)

    @staticmethod
    def split_text(
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        split_separator: str | None = None,
        chunk_mode: str = "paragraph",
    ) -> list[str]:
        """Split text using the current app splitter behavior."""
        return DocumentService._split_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_separator=split_separator or "\n\n",
            chunk_mode=chunk_mode,
        )

    @staticmethod
    def build_preview(
        document: Document,
        chunk_size: int,
        chunk_overlap: int,
        split_separator: str | None = None,
        chunk_mode: str = "paragraph",
        preview_limit: int = 20,
    ) -> dict:
        """Generate preview chunks without writing vectors."""
        text = DocumentParserService.extract_text(document)
        chunks = DocumentParserService.split_text(
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            split_separator=split_separator,
            chunk_mode=chunk_mode,
        )
        return {
            "chunk_count": len(chunks),
            "preview_chunks": [
                {
                    "chunk_index": index,
                    "content": chunk,
                }
                for index, chunk in enumerate(chunks[:preview_limit])
            ],
        }
