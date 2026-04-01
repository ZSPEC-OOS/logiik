"""
Logiik Retrieval Module.

Implements Retrieval-Augmented Generation (RAG) knowledge retrieval:
  1. Embed the query using SPECTER2.
  2. Search Pinecone (or FAISS fallback) for top-k nearest vectors.
  3. Check Redis cache for each result ID (cache hit → skip Firestore).
  4. Fetch full text from Firestore for cache misses.
  5. Populate cache with fetched text.
  6. Return ordered list of RetrievedChunk objects.

Cross-modal retrieval: supports both text queries and image queries.
Image queries embed via BLIP-2 caption embedding.

Usage:
    from logiik.retrieval.retrieve import Retriever
    retriever = Retriever()
    results = retriever.retrieve("enzyme folding at low pH", top_k=5)
    for r in results:
        print(r.text, r.score, r.metadata)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from PIL import Image

from logiik.storage.vector_db import VectorDB
from logiik.storage.text_store import TextStore
from logiik.storage.cache import Cache
from logiik.embeddings.embed import get_embedder
from logiik.utils.logging import get_logger

logger = get_logger("retrieval.retrieve")


# ─── Result type ─────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single retrieved knowledge chunk with full context.

    Attributes:
        id:        Unique chunk ID (matches vector DB and Firestore).
        text:      Full text content fetched from Firestore.
        score:     Cosine similarity score from vector search (0–1).
        metadata:  Metadata dict stored alongside the vector.
        source:    Source file or phase label.
        cache_hit: True if text was served from Redis cache.
    """
    id: str
    text: str
    score: float
    metadata: Dict = field(default_factory=dict)
    source: str = ""
    cache_hit: bool = False

    def __repr__(self):
        return (
            f"RetrievedChunk(id={self.id!r}, score={self.score:.4f}, "
            f"source={self.source!r}, cache_hit={self.cache_hit})"
        )


# ─── Retriever ────────────────────────────────────────────────────────────────

class Retriever:
    """
    Unified knowledge retriever combining vector search,
    Firestore full-text fetch, and Redis cache.

    Instantiate once and reuse — embedding model is lazy-loaded
    on first retrieve call.

    Args:
        vector_db:  VectorDB instance (created if not provided).
        text_store: TextStore instance (created if not provided).
        cache:      Cache instance (created if not provided).
    """

    def __init__(
        self,
        vector_db: Optional[VectorDB] = None,
        text_store: Optional[TextStore] = None,
        cache: Optional[Cache] = None,
    ):
        self._db = vector_db or VectorDB()
        self._store = text_store or TextStore()
        self._cache = cache or Cache()
        self._embedder = get_embedder()
        logger.info(
            f"Retriever initialised: backend={self._db.backend_name}, "
            f"cache_enabled={self._cache.is_enabled}"
        )

    # ─── Text retrieval ───────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
        min_score: float = 0.0
    ) -> List[RetrievedChunk]:
        """
        Retrieve top-k knowledge chunks relevant to a text query.

        Args:
            query:     Natural language query string.
            top_k:     Number of results to return.
            filter:    Optional Pinecone metadata filter
                       e.g. {"phase": {"$eq": "phase9"}}
            min_score: Minimum similarity score threshold (0–1).
                       Results below this are excluded.

        Returns:
            List of RetrievedChunk ordered by descending score.
        """
        if not query or not query.strip():
            logger.warning("retrieve() called with empty query.")
            return []

        logger.debug(f"Retrieving: query='{query[:80]}...', top_k={top_k}")

        # Step 1: Embed query
        query_emb = self._embedder.embed_text(query)

        # Step 2: Vector search
        matches = self._db.query(query_emb, top_k=top_k, filter=filter)

        if not matches:
            logger.debug("No vector matches returned.")
            return []

        # Step 3–5: Fetch text with cache
        results = []
        for match in matches:
            if match.score < min_score:
                continue
            chunk = self._fetch_chunk(match.id, match.score, match.metadata)
            if chunk is not None:
                results.append(chunk)

        logger.debug(
            f"Retrieved {len(results)} chunks "
            f"(cache hits: {sum(r.cache_hit for r in results)})"
        )
        return results

    # ─── Image retrieval ──────────────────────────────────────────────────

    def retrieve_by_image(
        self,
        image: Image.Image,
        top_k: int = 5,
        filter: Optional[Dict] = None,
        min_score: float = 0.0
    ) -> List[RetrievedChunk]:
        """
        Retrieve knowledge chunks relevant to an image query.
        Uses BLIP-2 to embed the image, then searches vector DB.

        Args:
            image:     PIL.Image.Image query image.
            top_k:     Number of results to return.
            filter:    Optional metadata filter.
            min_score: Minimum similarity score threshold.

        Returns:
            List of RetrievedChunk ordered by descending score.
        """
        logger.debug(f"Retrieving by image: size={image.size}, top_k={top_k}")

        # Embed image via BLIP-2
        image_emb = self._embedder.embed_image(image)

        # Vector search
        matches = self._db.query(image_emb, top_k=top_k, filter=filter)

        results = []
        for match in matches:
            if match.score < min_score:
                continue
            chunk = self._fetch_chunk(match.id, match.score, match.metadata)
            if chunk is not None:
                results.append(chunk)

        logger.debug(f"Retrieved {len(results)} chunks by image query.")
        return results

    # ─── Caption retrieval (text-based image search) ──────────────────────

    def retrieve_by_caption(
        self,
        caption: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
        min_score: float = 0.0
    ) -> List[RetrievedChunk]:
        """
        Retrieve image chunks by embedding their OCR-extracted caption.
        Faster than full BLIP-2 image embedding.

        Recommended for deduplication checks during Phase 8/9 ingestion.
        """
        caption_emb = self._embedder.embed_caption(caption)
        matches = self._db.query(caption_emb, top_k=top_k, filter=filter)

        results = []
        for match in matches:
            if match.score < min_score:
                continue
            chunk = self._fetch_chunk(match.id, match.score, match.metadata)
            if chunk is not None:
                results.append(chunk)

        return results

    # ─── Context assembly for RAG ─────────────────────────────────────────

    def build_context(
        self,
        query: str,
        top_k: int = 5,
        filter: Optional[Dict] = None,
        max_chars: int = 4000,
        separator: str = "\n\n---\n\n"
    ) -> str:
        """
        Retrieve top-k chunks and assemble into a single context string
        for injection into the language model prompt.

        Args:
            query:     Query string.
            top_k:     Number of chunks to retrieve.
            filter:    Optional metadata filter.
            max_chars: Maximum total characters in assembled context.
            separator: String placed between chunks.

        Returns:
            Assembled context string, truncated to max_chars.
        """
        chunks = self.retrieve(query, top_k=top_k, filter=filter)
        if not chunks:
            return ""

        parts = []
        total = 0
        for chunk in chunks:
            if not chunk.text:
                continue
            entry = f"[Source: {chunk.source}]\n{chunk.text}"
            if total + len(entry) > max_chars:
                # Truncate last entry to fit
                remaining = max_chars - total
                if remaining > 100:
                    parts.append(entry[:remaining] + "...")
                break
            parts.append(entry)
            total += len(entry)

        return separator.join(parts)

    # ─── Retrieval statistics ─────────────────────────────────────────────

    def stats(self) -> Dict:
        """
        Return retrieval layer statistics including vector DB stats
        and cache status.
        """
        return {
            "vector_db_backend": self._db.backend_name,
            "vector_db_stats": self._db.stats(),
            "cache_enabled": self._cache.is_enabled,
        }

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _fetch_chunk(
        self,
        chunk_id: str,
        score: float,
        metadata: Dict
    ) -> Optional[RetrievedChunk]:
        """
        Fetch full text for a single chunk ID.
        Check cache first; fall back to Firestore; populate cache on miss.

        Returns None if text cannot be retrieved from either source.
        """
        cache_hit = False

        # Cache lookup
        text = self._cache.get(chunk_id)
        if text:
            cache_hit = True
        else:
            # Firestore fetch
            text = self._store.fetch_chunk(chunk_id)
            if text:
                # Populate cache
                self._cache.set(chunk_id, text)
            else:
                # Text not found in either source
                # Use metadata text field as fallback if present
                text = metadata.get("text") or metadata.get("caption", "")
                if not text:
                    logger.warning(
                        f"No text found for chunk_id={chunk_id} "
                        "in cache, Firestore, or metadata."
                    )
                    return None

        return RetrievedChunk(
            id=chunk_id,
            text=text,
            score=score,
            metadata=metadata,
            source=metadata.get("source", ""),
            cache_hit=cache_hit
        )
