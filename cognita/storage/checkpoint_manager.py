"""
Local Knowledge Storage - Manages attachable knowledge base folder
with vector embeddings, model checkpoints, and training metadata.

Brain weights stay local. Metadata is optionally synced to Firebase
by passing a FirebaseMemory instance at construction time.
"""
import json
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import torch
from datetime import datetime

from cognita.storage.firebase_memory import FirebaseMemory


class KnowledgeBaseManager:
    """
    Manages the attachable knowledge_base/ folder structure:
    - embeddings/: Vector representations of learned knowledge
    - checkpoints/: Model snapshots at different training stages
    - training_data/: Historical training sessions
    - metadata/: Indices and configuration
    """

    def __init__(
        self,
        base_path: str = "./knowledge_base",
        firebase: Optional[FirebaseMemory] = None,
    ):
        self.base_path = Path(base_path)
        self.firebase = firebase  # None = local-only mode
        self._ensure_structure()

    def _ensure_structure(self):
        """Create knowledge base folder structure."""
        folders = ["embeddings", "checkpoints", "training_data", "metadata"]
        for folder in folders:
            (self.base_path / folder).mkdir(parents=True, exist_ok=True)

        # Initialize metadata index if not exists
        index_path = self.base_path / "metadata" / "knowledge_index.json"
        if not index_path.exists():
            self._save_index({
                "created": datetime.now().isoformat(),
                "checkpoints": [],
                "embeddings": [],
                "training_sessions": []
            })

    def _load_index(self) -> Dict:
        with open(self.base_path / "metadata" / "knowledge_index.json") as f:
            return json.load(f)

    def _save_index(self, index: Dict):
        with open(self.base_path / "metadata" / "knowledge_index.json", "w") as f:
            json.dump(index, f, indent=2)
        if self.firebase:
            self.firebase.sync_index(index)

    def save_checkpoint(
        self,
        model_state: Dict[str, torch.Tensor],
        training_stats: Dict[str, Any],
        name: Optional[str] = None
    ) -> Path:
        """Save model checkpoint to local knowledge base."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"checkpoint_{timestamp}"

        checkpoint_dir = self.base_path / "checkpoints" / name
        checkpoint_dir.mkdir(exist_ok=True)

        # Save model weights
        torch.save(model_state, checkpoint_dir / "model.pt")

        # Save training stats
        with open(checkpoint_dir / "stats.json", "w") as f:
            json.dump({
                **training_stats,
                "timestamp": timestamp,
                "checkpoint_name": name
            }, f, indent=2)

        # Update index
        index = self._load_index()
        index["checkpoints"].append({
            "name": name,
            "path": str(checkpoint_dir),
            "timestamp": timestamp,
            "stats": training_stats
        })
        self._save_index(index)

        if self.firebase:
            self.firebase.push_checkpoint(name, training_stats)

        print(f"Checkpoint saved: {checkpoint_dir}")
        return checkpoint_dir

    def load_checkpoint(self, name: str) -> Dict[str, torch.Tensor]:
        """Load model checkpoint from knowledge base."""
        checkpoint_path = self.base_path / "checkpoints" / name / "model.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No checkpoint found: {name}")

        return torch.load(checkpoint_path, map_location="cpu")

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        metadata: List[Dict],
        name: str
    ):
        """Save vector embeddings for RAG-style retrieval."""
        embed_dir = self.base_path / "embeddings" / name
        embed_dir.mkdir(exist_ok=True)

        # Save embeddings
        np.save(embed_dir / "vectors.npy", embeddings)

        # Save metadata
        with open(embed_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Update index
        index = self._load_index()
        index["embeddings"].append({
            "name": name,
            "path": str(embed_dir),
            "shape": list(embeddings.shape),
            "count": len(metadata)
        })
        self._save_index(index)

        if self.firebase:
            self.firebase.push_embedding_meta(name, list(embeddings.shape), len(metadata))

        print(f"Embeddings saved: {embed_dir}")

    def load_embeddings(self, name: str) -> tuple:
        """Load embeddings and their metadata."""
        embed_dir = self.base_path / "embeddings" / name

        if not embed_dir.exists():
            raise FileNotFoundError(f"No embeddings found: {name}")

        vectors = np.load(embed_dir / "vectors.npy")
        with open(embed_dir / "metadata.json") as f:
            metadata = json.load(f)

        return vectors, metadata

    def save_training_session(self, session_data: Dict[str, Any], name: Optional[str] = None):
        """Save a complete training session record."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = name or f"session_{timestamp}"

        session_path = self.base_path / "training_data" / f"{name}.json"
        with open(session_path, "w") as f:
            json.dump({**session_data, "timestamp": timestamp, "name": name}, f, indent=2)

        index = self._load_index()
        index["training_sessions"].append({
            "name": name,
            "path": str(session_path),
            "timestamp": timestamp
        })
        self._save_index(index)

        if self.firebase:
            self.firebase.push_session(name, session_data)

    def get_attachable_knowledge_summary(self) -> Dict:
        """Get summary of all knowledge that can be attached/loaded."""
        index = self._load_index()

        total_size = sum(
            f.stat().st_size
            for f in self.base_path.rglob("*")
            if f.is_file()
        ) / (1024 * 1024)  # MB

        return {
            "base_path": str(self.base_path.absolute()),
            "total_size_mb": round(total_size, 2),
            "checkpoints_count": len(index["checkpoints"]),
            "embeddings_count": len(index["embeddings"]),
            "training_sessions": len(index["training_sessions"]),
            "latest_checkpoint": index["checkpoints"][-1] if index["checkpoints"] else None,
            "available_checkpoints": [c["name"] for c in index["checkpoints"]]
        }

    def export_knowledge_package(self, output_path: str, name: str) -> Path:
        """Export entire knowledge base as attachable package."""
        output = Path(output_path) / f"nero_knowledge_{name}.zip"

        shutil.make_archive(
            str(output).replace(".zip", ""),
            "zip",
            self.base_path
        )

        print(f"Knowledge package exported: {output}")
        return output

    def import_knowledge_package(self, package_path: str):
        """Import knowledge package into local knowledge base."""
        with zipfile.ZipFile(package_path, "r") as zip_ref:
            zip_ref.extractall(self.base_path)

        print(f"Knowledge package imported to {self.base_path}")

    def prune_checkpoints(self, max_keep: int = 5):
        """Remove oldest checkpoints, keeping only max_keep most recent."""
        index = self._load_index()
        checkpoints = index["checkpoints"]

        if len(checkpoints) <= max_keep:
            return

        to_remove = checkpoints[:-max_keep]
        for cp in to_remove:
            cp_path = Path(cp["path"])
            if cp_path.exists():
                shutil.rmtree(cp_path)
                print(f"Pruned checkpoint: {cp['name']}")

        index["checkpoints"] = checkpoints[-max_keep:]
        self._save_index(index)
