"""
Shared model cache for embedding, reranker and chat client instances.
"""
from functools import lru_cache

import httpx
from pymilvus import MilvusClient, connections


@lru_cache(maxsize=4)
def get_ollama_http_client(base_url: str) -> httpx.Client:
    """Return a cached HTTP client for the local Ollama API."""
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
        # Ollama runs on localhost; do not route it through user/system proxies.
        trust_env=False,
    )


@lru_cache(maxsize=4)
def get_reranker(model_path: str):
    """Return a cached reranker tokenizer/model pair."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=4)
def get_http_client(base_url: str, api_key: str) -> httpx.Client:
    """Return a cached HTTP client for chat completion requests."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        headers=headers,
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=30.0),
        http2=False,
        # The app's configured Qwen endpoint is called directly. Ambient SOCKS
        # proxy variables require an optional socksio dependency and otherwise
        # prevent the client from being constructed.
        trust_env=False,
    )


@lru_cache(maxsize=4)
def get_milvus_client(uri: str, token: str) -> MilvusClient:
    """Return a cached Milvus client."""
    client = MilvusClient(uri=uri, token=token)
    connections._alias_handlers[client._using] = client._handler
    return client
