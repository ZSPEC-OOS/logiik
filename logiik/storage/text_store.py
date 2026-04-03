"""
Logiik Text Storage — Firebase REST API.

Extends the existing FirebaseMemory pattern from cognita/storage/firebase_memory.py.
Uses Firestore REST API directly — no Admin SDK, no service account required.
All binary data (embeddings, model weights) stays local or in S3.
Only structured text and metadata goes to Firestore.

Firestore layout (Logiik namespace):
  logiik/knowledge/{id}          — text chunks from ingestion
  logiik/phase7/{id}             — Phase 7 teacher/student QA pairs
  logiik/phase8/images/{id}      — Phase 8 image metadata
  logiik/phase9/chunks/{id}      — Phase 9 PDF text chunks
  logiik/memory/checkpoints/{id} — training checkpoints (migrated from NERO)
  logiik/memory/sessions/{id}    — training sessions
  logiik/memory/embeddings/{id}  — embedding metadata (not vectors)
"""
import os
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional

from logiik.config import CONFIG
from logiik.utils.logging import get_logger

logger = get_logger("storage.text_store")

# ─── Firestore value encoding/decoding ──────────────────────────────────────
# Preserved exactly from cognita/storage/firebase_memory.py

def _enc(value) -> dict:
    """Python value → Firestore typed field dict."""
    if value is None:
        return {"nullValue": None}
    if isinstance(value, bool):
        return {"booleanValue": value}
    if isinstance(value, int):
        return {"integerValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, list):
        return {"arrayValue": {"values": [_enc(v) for v in value]}}
    if isinstance(value, dict):
        return {"mapValue": {"fields": {k: _enc(v) for k, v in value.items()}}}
    return {"stringValue": str(value)}


def _dec(field: dict):
    """Firestore typed field dict → Python value."""
    if "nullValue"    in field: return None
    if "booleanValue" in field: return field["booleanValue"]
    if "integerValue" in field: return int(field["integerValue"])
    if "doubleValue"  in field: return field["doubleValue"]
    if "stringValue"  in field: return field["stringValue"]
    if "arrayValue"   in field:
        return [_dec(v) for v in field["arrayValue"].get("values", [])]
    if "mapValue"     in field:
        return {k: _dec(v) for k, v in field["mapValue"].get("fields", {}).items()}
    return None


def _doc_to_dict(doc: dict) -> dict:
    return {k: _dec(v) for k, v in doc.get("fields", {}).items()}


def _dict_to_body(data: dict) -> dict:
    return {"fields": {k: _enc(v) for k, v in data.items()}}


# ─── TextStore ───────────────────────────────────────────────────────────────

class TextStore:
    """
    Logiik text and metadata storage via Firestore REST API.

    Extends FirebaseMemory pattern with additional collections for
    Phase 7–9 data and knowledge chunk storage.

    Usage:
        from logiik.storage.text_store import TextStore
        store = TextStore()
        store.store_chunk("chunk_001", "enzyme folds at pH 4", {...})
        text = store.fetch_chunk("chunk_001")
    """

    _DEFAULT_PROJECT = "logiik"
    _DEFAULT_API_KEY = "AIzaSyDkbAhy7PlrYzHR5F-EDBquUtZ9fwLsyHg"

    def __init__(self):
        from pathlib import Path as _Path
        from dotenv import load_dotenv as _load_dotenv
        _load_dotenv(_Path(__file__).parent.parent.parent / ".env", override=True)
        self._project = (
            os.environ.get("FIREBASE_PROJECT")
            or CONFIG["firebase"].get("project")
            or self._DEFAULT_PROJECT
        )
        self._api_key = (
            os.environ.get("FIREBASE_API_KEY")
            or self._DEFAULT_API_KEY
        )
        self._base = (
            f"https://firestore.googleapis.com/v1"
            f"/projects/{self._project}/databases/(default)/documents"
        )
        self._params = {"key": self._api_key}
        logger.info(f"TextStore initialised: project={self._project}")

    # ─── Internal HTTP helpers (preserved from FirebaseMemory) ───────────

    def _sanitize(self, data: dict) -> dict:
        clean = {}
        for k, v in data.items():
            if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def _get(self, url: str) -> Optional[dict]:
        try:
            r = requests.get(url, params=self._params, timeout=10)
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.error(f"Firestore GET failed: {e}")
            return None

    def _patch(self, url: str, data: dict) -> bool:
        try:
            r = requests.patch(
                url, params=self._params,
                json=_dict_to_body(data), timeout=10
            )
            ok = r.status_code in (200, 201)
            if not ok:
                logger.warning(f"Firestore PATCH returned {r.status_code}")
            return ok
        except Exception as e:
            logger.error(f"Firestore PATCH failed: {e}")
            return False

    def _delete(self, url: str) -> bool:
        try:
            r = requests.delete(url, params=self._params, timeout=10)
            return r.status_code in (200, 204)
        except Exception as e:
            logger.error(f"Firestore DELETE failed: {e}")
            return False

    def _list(self, url: str) -> List[dict]:
        try:
            r = requests.get(url, params=self._params, timeout=10)
            if r.status_code != 200:
                return []
            return [_doc_to_dict(d) for d in r.json().get("documents", [])]
        except Exception as e:
            logger.error(f"Firestore LIST failed: {e}")
            return []

    # ─── Knowledge chunks (Phase 9 ingestion) ────────────────────────────

    def store_chunk(self, id: str, text: str, metadata: Dict) -> bool:
        """Store a text chunk from PDF ingestion."""
        url = f"{self._base}/logiik/knowledge/{id}"
        return self._patch(url, self._sanitize({
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata
        }))

    def fetch_chunk(self, id: str) -> Optional[str]:
        """Retrieve a text chunk by ID."""
        doc = self._get(f"{self._base}/logiik/knowledge/{id}")
        if not doc:
            return None
        data = _doc_to_dict(doc)
        return data.get("text")

    def delete_chunk(self, id: str) -> bool:
        return self._delete(f"{self._base}/logiik/knowledge/{id}")

    # ─── Phase 7 QA pairs ────────────────────────────────────────────────

    def store_phase7_teacher(self, question_id: str, data: Dict) -> bool:
        """Store a Phase 7 teacher-generated question + reasoning steps."""
        url = f"{self._base}/logiik/phase7/teacher/{question_id}"
        return self._patch(url, self._sanitize({
            **data,
            "timestamp": datetime.utcnow().isoformat()
        }))

    def store_phase7_student(self, question_id: str, data: Dict) -> bool:
        """Store a Phase 7 student attempt with correctness score."""
        url = f"{self._base}/logiik/phase7/student/{question_id}"
        return self._patch(url, self._sanitize({
            **data,
            "timestamp": datetime.utcnow().isoformat()
        }))

    def fetch_phase7_teacher(self, question_id: str) -> Optional[Dict]:
        doc = self._get(f"{self._base}/logiik/phase7/teacher/{question_id}")
        return _doc_to_dict(doc) if doc else None

    # ─── Phase 8 image metadata ───────────────────────────────────────────

    def store_image_metadata(self, image_id: str, metadata: Dict) -> bool:
        """Store Phase 8 scientific image metadata (not the image binary)."""
        url = f"{self._base}/logiik/phase8/images/{image_id}"
        return self._patch(url, self._sanitize({
            **metadata,
            "timestamp": datetime.utcnow().isoformat()
        }))

    def list_image_metadata(self) -> List[Dict]:
        return self._list(f"{self._base}/logiik/phase8/images")

    # ─── Checkpoints (migrated from FirebaseMemory) ───────────────────────

    def push_checkpoint(self, name: str, stats: Dict) -> bool:
        url = f"{self._base}/logiik/memory/checkpoints/{name}"
        return self._patch(url, self._sanitize({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            **stats
        }))

    def list_checkpoints(self) -> List[Dict]:
        docs = self._list(f"{self._base}/logiik/memory/checkpoints")
        return sorted(docs, key=lambda d: d.get("timestamp", ""))

    def delete_checkpoint(self, name: str) -> bool:
        return self._delete(f"{self._base}/logiik/memory/checkpoints/{name}")

    # ─── Training sessions ────────────────────────────────────────────────

    def push_session(self, name: str, session_data: Dict) -> bool:
        url = f"{self._base}/logiik/memory/sessions/{name}"
        return self._patch(url, self._sanitize({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            **session_data
        }))

    def list_sessions(self, limit: int = 50) -> List[Dict]:
        docs = self._list(f"{self._base}/logiik/memory/sessions")
        return sorted(
            docs, key=lambda d: d.get("timestamp", ""), reverse=True
        )[:limit]

    # ─── Embedding metadata ───────────────────────────────────────────────

    def push_embedding_meta(
        self, name: str, shape: List[int],
        count: int, extra: Dict = None
    ) -> bool:
        url = f"{self._base}/logiik/memory/embeddings/{name}"
        return self._patch(url, self._sanitize({
            "name": name,
            "shape": shape,
            "count": count,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {})
        }))

    def list_embeddings(self) -> List[Dict]:
        return self._list(f"{self._base}/logiik/memory/embeddings")

    # ─── Training Q&A records ─────────────────────────────────────────────

    def store_training_record(self, record_id: str, record: Dict) -> bool:
        """Sync one Q&A training record to Firestore immediately on generation."""
        url = f"{self._base}/training_records/{record_id}"
        return self._patch(url, self._sanitize({
            **record,
            "timestamp": datetime.utcnow().isoformat(),
        }))

    def list_training_records(self, limit: int = 500) -> List[Dict]:
        docs = self._list(f"{self._base}/training_records")
        return sorted(docs, key=lambda d: d.get("timestamp", ""))[:limit]

    # ─── Summary ─────────────────────────────────────────────────────────

    def get_summary(self) -> Dict:
        checkpoints = self.list_checkpoints()
        sessions = self.list_sessions()
        embeddings = self.list_embeddings()
        return {
            "checkpoints_count": len(checkpoints),
            "sessions_count": len(sessions),
            "embeddings_count": len(embeddings),
            "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        }
