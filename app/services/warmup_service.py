"""
Background warmup for heavy models and remote clients.
"""
from app.config import Config
from app.services.model_cache import get_http_client, get_milvus_client, get_reranker
from app.services.ollama_service import OllamaService


def warmup_models():
    """Preload local models and clients to reduce first-token latency."""
    try:
        OllamaService.embed_query("模型预热", model=Config.OLLAMA_EMBEDDING_MODEL)
    except Exception:
        pass

    try:
        get_reranker(Config.RERANK_MODEL_PATH)
    except Exception:
        pass

    try:
        get_milvus_client(Config.MILVUS_URI, Config.MILVUS_TOKEN)
    except Exception:
        pass

    if Config.LLM_API_KEY:
        try:
            get_http_client(
                base_url=Config.LLM_BASE_URL,
                api_key=Config.LLM_API_KEY,
            )
        except Exception:
            pass
