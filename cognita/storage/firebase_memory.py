"""
Firebase Memory Sync - Cloud persistence for Cognita AI memory.

What lives where:
  LOCAL  (knowledge_base/) - model weights (.pt), embedding vectors (.npy)
  FIREBASE (Firestore)     - knowledge index, training metadata, session history,
                             embedding metadata, checkpoint stats
"""
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore


class FirebaseMemory:
    """
    Syncs Cognita's memory metadata to Firestore.
    Binary brain data (weights, vectors) stays local — only
    lightweight structured memory goes to the cloud.

    Firestore collections:
      cognita/{brain_id}/index          - knowledge index document
      cognita/{brain_id}/checkpoints    - one doc per checkpoint
      cognita/{brain_id}/sessions       - one doc per training session
      cognita/{brain_id}/embeddings     - embedding metadata (not vectors)
    """

    def __init__(
        self,
        brain_id: str = "default",
        credential_path: Optional[str] = None,
    ):
        """
        Args:
            brain_id: Unique identifier for this brain instance.
                      Allows multiple brains in the same Firebase project.
            credential_path: Path to Firebase service account JSON.
                             Falls back to GOOGLE_APPLICATION_CREDENTIALS env var
                             or Application Default Credentials.
        """
        self.brain_id = brain_id

        if not firebase_admin._apps:
            cred = (
                credentials.Certificate(credential_path)
                if credential_path
                else credentials.ApplicationDefault()
            )
            firebase_admin.initialize_app(cred)

        self.db = firestore.client()
        self._root = self.db.collection("cognita").document(brain_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _col(self, name: str):
        return self._root.collection(name)

    def _sanitize(self, data: Dict) -> Dict:
        """Remove non-serializable values (tensors, ndarrays, Path objects)."""
        clean = {}
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    def sync_index(self, index: Dict[str, Any]):
        """Push the full knowledge index to Firestore."""
        self._root.set(self._sanitize({
            **index,
            "brain_id": self.brain_id,
            "last_synced": datetime.utcnow().isoformat(),
        }), merge=True)

    def get_index(self) -> Optional[Dict]:
        """Fetch the knowledge index from Firestore."""
        doc = self._root.get()
        return doc.to_dict() if doc.exists else None

    # ------------------------------------------------------------------
    # Checkpoints
    # ------------------------------------------------------------------

    def push_checkpoint(self, name: str, stats: Dict[str, Any]):
        """Record a checkpoint (stats only, not weights)."""
        self._col("checkpoints").document(name).set(self._sanitize({
            "name": name,
            "brain_id": self.brain_id,
            "timestamp": datetime.utcnow().isoformat(),
            **stats,
        }))

    def list_checkpoints(self) -> List[Dict]:
        """Return all checkpoint records ordered by timestamp."""
        docs = self._col("checkpoints").order_by("timestamp").stream()
        return [d.to_dict() for d in docs]

    def delete_checkpoint(self, name: str):
        """Remove a checkpoint record from Firestore."""
        self._col("checkpoints").document(name).delete()

    # ------------------------------------------------------------------
    # Training sessions
    # ------------------------------------------------------------------

    def push_session(self, name: str, session_data: Dict[str, Any]):
        """Record a training session."""
        self._col("sessions").document(name).set(self._sanitize({
            "name": name,
            "brain_id": self.brain_id,
            "timestamp": datetime.utcnow().isoformat(),
            **session_data,
        }))

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        """Return recent training sessions."""
        docs = (
            self._col("sessions")
            .order_by("timestamp", direction=firestore.Query.DESCENDING)
            .limit(limit)
            .stream()
        )
        return [d.to_dict() for d in docs]

    # ------------------------------------------------------------------
    # Embedding metadata (vectors stay local)
    # ------------------------------------------------------------------

    def push_embedding_meta(self, name: str, shape: List[int], count: int, extra: Dict = None):
        """Record embedding metadata — vectors themselves stay local."""
        self._col("embeddings").document(name).set(self._sanitize({
            "name": name,
            "brain_id": self.brain_id,
            "shape": shape,
            "count": count,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {}),
        }))

    def list_embeddings(self) -> List[Dict]:
        """Return all embedding metadata records."""
        docs = self._col("embeddings").stream()
        return [d.to_dict() for d in docs]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_summary(self) -> Dict:
        """High-level memory summary from Firestore."""
        checkpoints = self.list_checkpoints()
        sessions = self.list_sessions()
        embeddings = self.list_embeddings()

        return {
            "brain_id": self.brain_id,
            "checkpoints_count": len(checkpoints),
            "sessions_count": len(sessions),
            "embeddings_count": len(embeddings),
            "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        }
