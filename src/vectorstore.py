"""
vectorstore.py — ChromaDB vector store manager.

Responsibilities:
  • Add document chunks with deduplication
  • Similarity search with metadata filters
  • Collection stats & document listing
  • Delete documents by source
  • Persistent storage via HuggingFace dataset (push/pull)
"""
from __future__ import annotations
import hashlib
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import settings
from src.embeddings import get_embeddings
from src.logger import logger


class VectorStoreManager:
    """Manages the ChromaDB vector store for the research assistant."""

    def __init__(self):
        self._persist_dir = str(Path(settings.chroma_persist_dir).resolve())
        self._collection_name = settings.chroma_collection
        self._embeddings = get_embeddings()
        self._store: Optional[Chroma] = None
        self._init_store()

    def _init_store(self):
        """Initialise or load the ChromaDB store. Pulls from HF on first run."""
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)

        # Pull from HuggingFace persistent storage if available
        try:
            from src.storage import pull_vectorstore, is_persistent_storage_enabled
            if is_persistent_storage_enabled():
                pull_vectorstore(self._persist_dir)
        except Exception as e:
            logger.warning(f"Storage pull failed (non-fatal): {e}")

        try:
            self._store = Chroma(
                collection_name=self._collection_name,
                embedding_function=self._embeddings,
                persist_directory=self._persist_dir,
            )
            count = self._safe_count()
            logger.info(
                f"VectorStore ready — collection='{self._collection_name}' "
                f"docs={count} persist='{self._persist_dir}'"
            )
        except Exception as e:
            logger.warning(f"VectorStore init failed ({e}) — resetting.")
            self._reset_store()

    def _reset_store(self):
        """Wipe and recreate the vector store from scratch."""
        try:
            if Path(self._persist_dir).exists():
                shutil.rmtree(self._persist_dir)
                logger.info(f"Wiped corrupt DB at '{self._persist_dir}'")
        except Exception as e:
            logger.error(f"Could not wipe DB dir: {e}")

        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._store = Chroma(
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            persist_directory=self._persist_dir,
        )
        logger.info("VectorStore reset — fresh empty collection created")

    def _safe_count(self) -> int:
        """
        Safely get document count.
        Handles chromadb 0.5.x _decode_seq_id bug gracefully.
        """
        try:
            return self._store._collection.count()
        except TypeError:
            try:
                result = self._store._collection.get(include=[])
                return len(result.get("ids", []))
            except Exception:
                return 0
        except Exception:
            return 0

    def _push_to_storage(self):
        """Push vectorstore to HuggingFace after every write operation."""
        try:
            from src.storage import push_vectorstore, is_persistent_storage_enabled
            if is_persistent_storage_enabled():
                push_vectorstore(self._persist_dir)
        except Exception as e:
            logger.warning(f"Storage push failed (non-fatal): {e}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_id(doc: Document) -> str:
        """Stable ID based on source + content hash."""
        raw = f"{doc.metadata.get('source', '')}::{doc.page_content[:200]}"
        return hashlib.md5(raw.encode()).hexdigest()

    # ── Write ops ─────────────────────────────────────────────────────────────

    def add_documents(self, docs: List[Document]) -> Tuple[int, int]:
        """
        Add documents to the store with deduplication.
        Returns (added, skipped) counts.
        """
        if not docs:
            return 0, 0

        existing_ids: set = set()
        try:
            existing = self._store._collection.get(include=[])
            existing_ids = set(existing.get("ids", []))
        except Exception:
            pass

        new_docs, new_ids, skipped = [], [], 0
        for doc in docs:
            doc_id = self._chunk_id(doc)
            if doc_id in existing_ids:
                skipped += 1
            else:
                new_docs.append(doc)
                new_ids.append(doc_id)

        if new_docs:
            self._store.add_documents(new_docs, ids=new_ids)
            logger.info(f"Added {len(new_docs)} chunks | Skipped {skipped} duplicates")
            # Push to persistent storage after every successful write
            self._push_to_storage()

        return len(new_docs), skipped

    # ── Read ops ──────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        k: int = None,
        source_filter: Optional[str] = None,
    ) -> List[Document]:
        """Similarity search. Optionally filter by source name."""
        k = k or settings.top_k_results
        search_kwargs: dict = {"k": k}
        if source_filter:
            search_kwargs["filter"] = {"source": source_filter}
        results = self._store.similarity_search(query, **search_kwargs)
        logger.debug(f"Search '{query[:60]}' → {len(results)} results")
        return results

    def search_with_scores(
        self,
        query: str,
        k: int = None,
    ) -> List[Tuple[Document, float]]:
        """Similarity search returning (document, score) tuples."""
        k = k or settings.top_k_results
        results = self._store.similarity_search_with_relevance_scores(query, k=k)
        return results

    # ── Stats & metadata ──────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        """Return collection statistics."""
        try:
            total = self._safe_count()
            all_meta = self._store._collection.get(include=["metadatas"])
            sources: Dict[str, Dict] = {}
            for meta in (all_meta.get("metadatas") or []):
                if not meta:
                    continue
                src = meta.get("source", "unknown")
                if src not in sources:
                    sources[src] = {
                        "source": src,
                        "type": meta.get("source_type", "unknown"),
                        "ingested_at": meta.get("ingested_at", ""),
                        "chunks": 0,
                    }
                sources[src]["chunks"] += 1
            return {
                "total_chunks": total,
                "total_sources": len(sources),
                "sources": list(sources.values()),
            }
        except Exception as e:
            logger.error(f"Stats error: {e}")
            return {"total_chunks": 0, "total_sources": 0, "sources": []}

    def list_sources(self) -> List[str]:
        """Return all unique source names."""
        stats = self.get_stats()
        return [s["source"] for s in stats["sources"]]

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a specific source. Returns deleted count."""
        try:
            results = self._store._collection.get(
                where={"source": source}, include=[]
            )
            ids = results.get("ids", [])
            if ids:
                self._store._collection.delete(ids=ids)
                logger.info(f"Deleted {len(ids)} chunks from source='{source}'")
                # Push to persistent storage after delete
                self._push_to_storage()
            return len(ids)
        except Exception as e:
            logger.error(f"Delete error for source '{source}': {e}")
            return 0

    def is_empty(self) -> bool:
        return self._safe_count() == 0

    def as_retriever(self, k: int = None):
        """Return a LangChain retriever interface."""
        return self._store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k or settings.top_k_results},
        )


# Singleton
_vsm_instance: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    global _vsm_instance
    if _vsm_instance is None:
        _vsm_instance = VectorStoreManager()
    return _vsm_instance