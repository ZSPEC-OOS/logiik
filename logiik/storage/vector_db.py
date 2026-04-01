"""
Logiik Vector Database Abstraction.

Provides a unified interface over Pinecone (primary) and FAISS (local
fallback). Backend is selected via config.yaml: vector_db.backend.

All other modules call this interface exclusively — no direct
Pinecone or FAISS imports elsewhere in the codebase.
"""
import os
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from logiik.config import CONFIG
from logiik.utils.logging import get_logger

logger = get_logger("storage.vector_db")


# ─── Unified result type ─────────────────────────────────────────────────────

class VectorMatch:
    """Single result from a vector similarity query."""
    def __init__(self, id: str, score: float, metadata: Dict):
        self.id = id
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        return f"VectorMatch(id={self.id!r}, score={self.score:.4f})"


# ─── Backend: Pinecone ───────────────────────────────────────────────────────

class PineconeBackend:
    """
    Pinecone vector database backend.
    Uses pinecone-client v3+ (Pinecone class, not pinecone.init).
    """

    def __init__(self):
        api_key = os.environ.get("PINECONE_API_KEY")
        host = os.environ.get("PINECONE_HOST")

        if not api_key:
            raise EnvironmentError(
                "PINECONE_API_KEY not set. Add it to your .env file."
            )

        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        self._index = pc.Index(host=host)
        logger.info(f"Pinecone backend initialised: host={host}")

    def upsert(self, id: str, embedding: np.ndarray, metadata: Dict) -> bool:
        """Insert or update a single vector with metadata."""
        try:
            self._index.upsert(vectors=[{
                "id": id,
                "values": embedding.tolist(),
                "metadata": metadata
            }])
            logger.debug(f"Upserted vector id={id}")
            return True
        except Exception as e:
            logger.error(f"Pinecone upsert failed for id={id}: {e}")
            return False

    def upsert_batch(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict],
        batch_size: int = 100
    ) -> int:
        """
        Batch upsert for large datasets.
        Returns count of successfully upserted vectors.
        """
        success = 0
        vectors = [
            {"id": i, "values": e.tolist(), "metadata": m}
            for i, e, m in zip(ids, embeddings, metadatas)
        ]
        for start in range(0, len(vectors), batch_size):
            batch = vectors[start:start + batch_size]
            try:
                self._index.upsert(vectors=batch)
                success += len(batch)
                logger.debug(f"Batch upserted {len(batch)} vectors")
            except Exception as e:
                logger.error(f"Batch upsert failed at offset {start}: {e}")
        return success

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[VectorMatch]:
        """
        Query top_k nearest vectors.
        Optional metadata filter e.g. {"phase": {"$eq": "phase7"}}
        """
        try:
            kwargs = {
                "vector": embedding.tolist(),
                "top_k": top_k,
                "include_metadata": True
            }
            if filter:
                kwargs["filter"] = filter
            result = self._index.query(**kwargs)
            return [
                VectorMatch(
                    id=m["id"],
                    score=m["score"],
                    metadata=m.get("metadata", {})
                )
                for m in result.get("matches", [])
            ]
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            return []

    def delete(self, id: str) -> bool:
        """Delete a single vector by ID."""
        try:
            self._index.delete(ids=[id])
            logger.debug(f"Deleted vector id={id}")
            return True
        except Exception as e:
            logger.error(f"Pinecone delete failed for id={id}: {e}")
            return False

    def stats(self) -> Dict:
        """Return index statistics."""
        try:
            return self._index.describe_index_stats()
        except Exception as e:
            logger.error(f"Pinecone stats failed: {e}")
            return {}


# ─── Backend: FAISS (local fallback) ────────────────────────────────────────

class FAISSBackend:
    """
    Local FAISS vector database.
    Used as fallback when Pinecone is unavailable or for offline dev.
    Stores metadata in a parallel dict since FAISS is index-only.
    """

    def __init__(self):
        try:
            import faiss
            self._faiss = faiss
        except ImportError:
            raise ImportError(
                "faiss-cpu not installed. Run: pip install faiss-cpu"
            )

        cfg = CONFIG["vector_db"]["faiss"]
        self._dim = cfg["dimension"]
        self._index_path = Path(cfg["index_path"])
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

        self._meta_path = self._index_path.parent / "faiss_metadata.npy"
        self._id_path = self._index_path.parent / "faiss_ids.npy"

        # Load or create index
        if self._index_path.exists():
            self._index = self._faiss.read_index(str(self._index_path))
            self._metadata = list(np.load(self._meta_path, allow_pickle=True))
            self._ids = list(np.load(self._id_path, allow_pickle=True))
            logger.info(f"FAISS index loaded: {len(self._ids)} vectors")
        else:
            self._index = self._faiss.IndexFlatIP(self._dim)  # Inner product
            self._metadata: List[Dict] = []
            self._ids: List[str] = []
            logger.info("FAISS index created (empty)")

    def _save(self):
        self._faiss.write_index(self._index, str(self._index_path))
        np.save(self._meta_path, np.array(self._metadata, dtype=object))
        np.save(self._id_path, np.array(self._ids, dtype=object))

    def upsert(self, id: str, embedding: np.ndarray, metadata: Dict) -> bool:
        try:
            vec = embedding.astype(np.float32).reshape(1, -1)
            self._faiss.normalize_L2(vec)
            self._index.add(vec)
            self._ids.append(id)
            self._metadata.append(metadata)
            self._save()
            return True
        except Exception as e:
            logger.error(f"FAISS upsert failed: {e}")
            return False

    def upsert_batch(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict],
        batch_size: int = 100
    ) -> int:
        try:
            vecs = np.array(embeddings, dtype=np.float32)
            self._faiss.normalize_L2(vecs)
            self._index.add(vecs)
            self._ids.extend(ids)
            self._metadata.extend(metadatas)
            self._save()
            return len(ids)
        except Exception as e:
            logger.error(f"FAISS batch upsert failed: {e}")
            return 0

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[VectorMatch]:
        try:
            vec = embedding.astype(np.float32).reshape(1, -1)
            self._faiss.normalize_L2(vec)
            scores, indices = self._index.search(vec, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or idx >= len(self._ids):
                    continue
                meta = self._metadata[idx]
                # Optional metadata filter
                if filter:
                    match = all(
                        meta.get(k) == v.get("$eq", v)
                        for k, v in filter.items()
                    )
                    if not match:
                        continue
                results.append(VectorMatch(
                    id=self._ids[idx],
                    score=float(score),
                    metadata=meta
                ))
            return results
        except Exception as e:
            logger.error(f"FAISS query failed: {e}")
            return []

    def delete(self, id: str) -> bool:
        # FAISS IndexFlatIP does not support deletion natively
        # Mark as deleted in metadata
        if id in self._ids:
            idx = self._ids.index(id)
            self._metadata[idx]["_deleted"] = True
            self._save()
            logger.debug(f"FAISS: marked id={id} as deleted")
            return True
        return False

    def stats(self) -> Dict:
        return {
            "total_vector_count": len(self._ids),
            "dimension": self._dim,
            "backend": "faiss"
        }


# ─── Unified VectorDB interface ──────────────────────────────────────────────

class VectorDB:
    """
    Unified vector database interface.
    Backend selected from config.yaml: vector_db.backend (pinecone | faiss).

    Usage:
        from logiik.storage.vector_db import VectorDB
        db = VectorDB()
        db.upsert(id, embedding, metadata)
        matches = db.query(embedding, top_k=5)
    """

    def __init__(self, backend: Optional[str] = None):
        cfg_backend = backend or CONFIG["vector_db"]["backend"]

        if cfg_backend == "pinecone":
            try:
                self._backend = PineconeBackend()
                self._backend_name = "pinecone"
            except Exception as e:
                logger.warning(
                    f"Pinecone init failed ({e}). Falling back to FAISS."
                )
                self._backend = FAISSBackend()
                self._backend_name = "faiss_fallback"
        elif cfg_backend == "faiss":
            self._backend = FAISSBackend()
            self._backend_name = "faiss"
        else:
            raise ValueError(f"Unknown vector_db backend: {cfg_backend}")

        logger.info(f"VectorDB initialised with backend: {self._backend_name}")

    def upsert(self, id: str, embedding: np.ndarray, metadata: Dict) -> bool:
        return self._backend.upsert(id, embedding, metadata)

    def upsert_batch(
        self,
        ids: List[str],
        embeddings: List[np.ndarray],
        metadatas: List[Dict],
        batch_size: int = 100
    ) -> int:
        return self._backend.upsert_batch(ids, embeddings, metadatas, batch_size)

    def query(
        self,
        embedding: np.ndarray,
        top_k: int = 5,
        filter: Optional[Dict] = None
    ) -> List[VectorMatch]:
        return self._backend.query(embedding, top_k, filter)

    def delete(self, id: str) -> bool:
        return self._backend.delete(id)

    def stats(self) -> Dict:
        return self._backend.stats()

    @property
    def backend_name(self) -> str:
        return self._backend_name
