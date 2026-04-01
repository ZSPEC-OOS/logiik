"""
Firebase Memory Sync - Cloud persistence for NERO memory.

Uses the Firestore REST API — no service account or SDK init required.
Only lightweight structured metadata goes to the cloud; binary data
(model weights, vectors) stays in the local knowledge-base folder.

Firestore layout:
  nero/memory                     — root index document
  nero/memory/checkpoints/{name}  — one doc per checkpoint
  nero/memory/sessions/{name}     — one doc per training session
  nero/memory/embeddings/{name}   — embedding metadata (not vectors)
"""
import requests
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Firebase project config ────────────────────────────────────────────
_PROJECT  = "nero-85ed0"
_API_KEY  = "AIzaSyAD1Lu8bT5VTZC2k4suWk_X2FfSW9H-fUI"
_BASE_URL = (
    f"https://firestore.googleapis.com/v1"
    f"/projects/{_PROJECT}/databases/(default)/documents"
)


# ── Firestore value encoding/decoding ─────────────────────────────────

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


class FirebaseMemory:
    """Syncs NERO memory metadata to Firestore via REST API."""

    _ROOT = f"{_BASE_URL}/nero/memory"

    def __init__(self):
        self._p = {"key": _API_KEY}  # query-string params for all requests

    # ── Internal helpers ───────────────────────────────────────────────

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
            r = requests.get(url, params=self._p, timeout=10)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def _patch(self, url: str, data: dict) -> bool:
        try:
            r = requests.patch(url, params=self._p,
                               json=_dict_to_body(data), timeout=10)
            return r.status_code in (200, 201)
        except Exception:
            return False

    def _delete(self, url: str) -> bool:
        try:
            r = requests.delete(url, params=self._p, timeout=10)
            return r.status_code in (200, 204)
        except Exception:
            return False

    def _list(self, url: str) -> List[dict]:
        try:
            r = requests.get(url, params=self._p, timeout=10)
            if r.status_code != 200:
                return []
            return [_doc_to_dict(d) for d in r.json().get("documents", [])]
        except Exception:
            return []

    # ── Index ──────────────────────────────────────────────────────────

    def sync_index(self, index: dict):
        self._patch(self._ROOT, self._sanitize({
            **index,
            "last_synced": datetime.utcnow().isoformat(),
        }))

    def get_index(self) -> Optional[dict]:
        doc = self._get(self._ROOT)
        return _doc_to_dict(doc) if doc else None

    # ── Checkpoints ────────────────────────────────────────────────────

    def push_checkpoint(self, name: str, stats: dict):
        self._patch(f"{self._ROOT}/checkpoints/{name}", self._sanitize({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            **stats,
        }))

    def list_checkpoints(self) -> List[dict]:
        docs = self._list(f"{self._ROOT}/checkpoints")
        return sorted(docs, key=lambda d: d.get("timestamp", ""))

    def delete_checkpoint(self, name: str):
        self._delete(f"{self._ROOT}/checkpoints/{name}")

    # ── Training sessions ──────────────────────────────────────────────

    def push_session(self, name: str, session_data: dict):
        self._patch(f"{self._ROOT}/sessions/{name}", self._sanitize({
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            **session_data,
        }))

    def list_sessions(self, limit: int = 50) -> List[dict]:
        docs = self._list(f"{self._ROOT}/sessions")
        return sorted(docs, key=lambda d: d.get("timestamp", ""), reverse=True)[:limit]

    # ── Embedding metadata ─────────────────────────────────────────────

    def push_embedding_meta(self, name: str, shape: List[int], count: int,
                             extra: dict = None):
        self._patch(f"{self._ROOT}/embeddings/{name}", self._sanitize({
            "name": name,
            "shape": shape,
            "count": count,
            "timestamp": datetime.utcnow().isoformat(),
            **(extra or {}),
        }))

    def list_embeddings(self) -> List[dict]:
        return self._list(f"{self._ROOT}/embeddings")

    # ── Summary ────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        checkpoints = self.list_checkpoints()
        sessions    = self.list_sessions()
        embeddings  = self.list_embeddings()
        return {
            "checkpoints_count": len(checkpoints),
            "sessions_count":    len(sessions),
            "embeddings_count":  len(embeddings),
            "latest_checkpoint": checkpoints[-1] if checkpoints else None,
        }
