"""
config.py — Centralised configuration loader.
Reads from .env and exposes a typed Settings object.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    import streamlit as st
    for k, v in st.secrets.items():
        os.environ.setdefault(k, str(v))
except Exception:
    pass

# Load .env from project root
load_dotenv(Path(__file__).parent.parent / ".env")


class Settings(BaseModel):
    # LLM
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    # ✅ gemini-1.5-flash removed from API — gemini-2.0-flash is the correct default
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.0-flash"))

    # Embeddings
    embedding_provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))

    # Vector store
    chroma_persist_dir: str = Field(default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", "./vectorstore"))
    chroma_collection: str = Field(default_factory=lambda: os.getenv("CHROMA_COLLECTION", "research_docs"))

    # RAG
    chunk_size: int = Field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "800")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "150")))
    top_k_results: int = Field(default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "6")))
    max_sources: int = Field(default_factory=lambda: int(os.getenv("MAX_SOURCES_PER_QUERY", "5")))
    temperature: float = Field(default_factory=lambda: float(os.getenv("TEMPERATURE", "0.2")))

    # App
    app_title: str = Field(default_factory=lambda: os.getenv("APP_TITLE", "AI Research Assistant"))
    upload_dir: Path = Path("./uploads")
    log_file: str = Field(default_factory=lambda: os.getenv("LOG_FILE", "./logs/app.log"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    def validate_keys(self) -> list[str]:
        """Return list of missing required keys based on active provider."""
        issues = []
        if self.llm_provider == "openai" and not self.openai_api_key.startswith("sk-"):
            issues.append("OPENAI_API_KEY is missing or invalid in .env")
        if self.llm_provider == "groq" and not self.groq_api_key.startswith("gsk_"):
            issues.append("GROQ_API_KEY is missing or invalid in .env")
        if self.llm_provider == "gemini" and not self.gemini_api_key:
            issues.append("GEMINI_API_KEY is missing in .env")
        return issues


settings = Settings()