"""
embeddings.py — Embedding model factory.
Supports local sentence-transformers (free) and OpenAI embeddings.
"""
from __future__ import annotations
from functools import lru_cache
from langchain_core.embeddings import Embeddings
from src.config import settings
from src.logger import logger


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return a cached embedding model."""
    provider = settings.embedding_provider.lower()
    logger.info(f"Initialising embeddings — provider={provider}")

    if provider == "local":
        # Detect correct device safely — fixes Streamlit Cloud crash
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Local embeddings ready — model={settings.embedding_model} device={device}")
        return embeddings

    elif provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=settings.openai_api_key,
        )
        logger.info("OpenAI embeddings ready — model=text-embedding-3-small")
        return embeddings

    else:
        raise ValueError(
            f"Unknown EMBEDDING_PROVIDER='{provider}'. Choose 'local' or 'openai'."
        )
