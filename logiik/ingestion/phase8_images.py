"""
Logiik Phase 8 — Scientific Image Analysis.

Extracts, classifies, embeds, and deduplicates scientific images
from local folders or PDFs before Phase 9 ingestion.

Pipeline:
  1. Extract images from folder (PNG/JPG/JPEG) or PDF pages.
  2. OCR-extract captions from each image via Tesseract.
  3. Classify image type (microscopy, diagram, plot, etc.).
  4. Compute BLIP-2 embedding (falls back to SPECTER2 caption
     embedding if BLIP-2 unavailable on CPU).
  5. Deduplicate against phase8_image_db.
  6. Store metadata in Firebase + embedding in Pinecone.
  7. Return new embeddings for downstream Phase 9 dedup check.

Phase 9 (PDF ingestion) calls is_image_duplicate() before
embedding any PDF-extracted image to avoid redundant storage.

Usage:
    from logiik.ingestion.phase8_images import Phase8ImagePipeline
    pipeline = Phase8ImagePipeline()
    pipeline.process_folder("path/to/image/folder")
    pipeline.process_pdf_images(pdf_path, page_images)
    # In Phase 9:
    if not pipeline.is_image_duplicate(new_image):
        pipeline.add_image(new_image, caption, source)
"""
import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

from logiik.config import CONFIG
from logiik.embeddings.embed import get_embedder
from logiik.storage.vector_db import VectorDB
from logiik.storage.text_store import TextStore
from logiik.storage.cache import Cache
from logiik.utils.helpers import is_duplicate
from logiik.utils.logging import get_logger

logger = get_logger("ingestion.phase8_images")

# ─── Image type categories ────────────────────────────────────────────────────

IMAGE_TYPES = CONFIG.get("ingestion", {}).get(
    "image_types",
    ["microscopy", "diagram", "plot", "chart", "chemical_structure", "other"]
)

DEDUP_THRESHOLD = CONFIG.get("ingestion", {}).get(
    "deduplication_threshold", 0.90
)


# ─── Image record ─────────────────────────────────────────────────────────────

class ImageRecord:
    """
    Metadata record for a single processed scientific image.
    Stored in Firebase; embedding stored in Pinecone.
    """
    def __init__(
        self,
        image_id: str,
        image_path: str,
        caption: str,
        embedding: np.ndarray,
        image_type: str,
        source: str,
        page_number: Optional[int] = None,
    ):
        self.image_id = image_id
        self.image_path = image_path
        self.caption = caption
        self.embedding = embedding
        self.image_type = image_type
        self.source = source
        self.page_number = page_number

    def to_metadata(self) -> Dict:
        """Serialisable metadata dict for Firebase + Pinecone."""
        return {
            "image_id": self.image_id,
            "image_path": self.image_path,
            "caption": self.caption[:500],   # Firestore field limit safety
            "image_type": self.image_type,
            "source": self.source,
            "page_number": self.page_number or -1,
            "phase": "phase8",
        }

    def __repr__(self):
        return (
            f"ImageRecord(id={self.image_id[:8]}..., "
            f"type={self.image_type}, source={self.source!r})"
        )


# ─── Phase 8 Pipeline ─────────────────────────────────────────────────────────

class Phase8ImagePipeline:
    """
    Full Phase 8 scientific image analysis pipeline.

    Maintains an in-memory phase8_image_db (list of ImageRecord)
    mirrored to Firebase + Pinecone for persistence.

    Thread safety: not thread-safe. Use one instance per process.
    """

    def __init__(
        self,
        vector_db: Optional[VectorDB] = None,
        text_store: Optional[TextStore] = None,
        cache: Optional[Cache] = None,
        dedup_threshold: float = DEDUP_THRESHOLD,
    ):
        self._db = vector_db or VectorDB()
        self._store = text_store or TextStore()
        self._cache = cache or Cache()
        self._embedder = get_embedder()
        self._threshold = dedup_threshold

        # In-memory image database — embeddings kept for fast dedup
        self.phase8_image_db: List[ImageRecord] = []

        logger.info(
            f"Phase8ImagePipeline initialised: "
            f"dedup_threshold={self._threshold}"
        )

    # ─── Public entry points ──────────────────────────────────────────────

    def process_folder(self, folder_path: str) -> Dict:
        """
        Process all images in a local folder.

        Args:
            folder_path: Path to folder containing PNG/JPG/JPEG files.

        Returns:
            Dict with counts: processed, new, duplicates, errors.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        image_files = [
            f for f in folder.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ]

        logger.info(
            f"Phase 8: processing folder '{folder_path}': "
            f"{len(image_files)} images found."
        )

        stats = {"processed": 0, "new": 0, "duplicates": 0, "errors": 0}

        for img_path in image_files:
            try:
                img = Image.open(img_path)
                caption = self._ocr_caption(img)
                added = self.add_image(
                    image=img,
                    caption=caption,
                    source=str(img_path),
                    image_path=str(img_path),
                )
                stats["processed"] += 1
                if added:
                    stats["new"] += 1
                else:
                    stats["duplicates"] += 1
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Phase 8 folder complete: {stats}"
        )
        return stats

    def process_pdf_images(
        self,
        pdf_path: str,
        page_images: List[Tuple[Image.Image, int]],
    ) -> Dict:
        """
        Process images extracted from a PDF by Phase 9.

        Called by Phase9PDFPipeline before embedding PDF images,
        to register them in phase8_image_db and skip duplicates.

        Args:
            pdf_path:    Source PDF file path (for metadata).
            page_images: List of (PIL.Image, page_number) tuples.

        Returns:
            Dict with counts: processed, new, duplicates, errors.
        """
        stats = {"processed": 0, "new": 0, "duplicates": 0, "errors": 0}

        for img, page_num in page_images:
            try:
                caption = self._ocr_caption(img)
                added = self.add_image(
                    image=img,
                    caption=caption,
                    source=pdf_path,
                    image_path=f"{pdf_path}::page_{page_num}",
                    page_number=page_num,
                )
                stats["processed"] += 1
                if added:
                    stats["new"] += 1
                else:
                    stats["duplicates"] += 1
            except Exception as e:
                logger.error(
                    f"Error processing PDF image page {page_num}: {e}"
                )
                stats["errors"] += 1

        return stats

    def add_image(
        self,
        image: Image.Image,
        caption: str,
        source: str,
        image_path: str = "",
        page_number: Optional[int] = None,
    ) -> bool:
        """
        Add a single image to phase8_image_db if not duplicate.

        Args:
            image:       PIL.Image.Image.
            caption:     OCR-extracted or provided caption text.
            source:      Source file path or identifier.
            image_path:  Full path to image file (if available).
            page_number: PDF page number (if extracted from PDF).

        Returns:
            True if image was new and added.
            False if duplicate — image skipped.
        """
        # Compute embedding
        emb = self._compute_embedding(image, caption)

        # Deduplication check against existing phase8_image_db
        existing_embs = [r.embedding for r in self.phase8_image_db]
        if is_duplicate(emb, existing_embs, threshold=self._threshold):
            logger.debug(
                f"Phase 8: duplicate image skipped: source={source}"
            )
            return False

        # Classify image type
        image_type = self._classify_image_type(caption)

        # Create record
        image_id = str(uuid.uuid4())
        record = ImageRecord(
            image_id=image_id,
            image_path=image_path or source,
            caption=caption,
            embedding=emb,
            image_type=image_type,
            source=source,
            page_number=page_number,
        )
        self.phase8_image_db.append(record)

        # Persist to Pinecone
        self._db.upsert(
            id=f"phase8_{image_id}",
            embedding=emb,
            metadata=record.to_metadata()
        )

        # Persist metadata to Firebase
        self._store.store_image_metadata(image_id, record.to_metadata())

        logger.debug(
            f"Phase 8: new image added: id={image_id[:8]}, "
            f"type={image_type}, source={source}"
        )
        return True

    def is_image_duplicate(
        self,
        image: Image.Image,
        caption: Optional[str] = None,
    ) -> bool:
        """
        Check if an image is already in phase8_image_db.
        Called by Phase 9 before embedding PDF images.

        Args:
            image:   PIL.Image.Image to check.
            caption: Optional pre-extracted caption. If None,
                     OCR is run automatically.

        Returns:
            True if duplicate exists.
        """
        if not self.phase8_image_db:
            return False
        cap = caption or self._ocr_caption(image)
        emb = self._compute_embedding(image, cap)
        existing_embs = [r.embedding for r in self.phase8_image_db]
        return is_duplicate(emb, existing_embs, threshold=self._threshold)

    def get_stats(self) -> Dict:
        """Return Phase 8 pipeline statistics."""
        type_counts: Dict[str, int] = {}
        for record in self.phase8_image_db:
            type_counts[record.image_type] = (
                type_counts.get(record.image_type, 0) + 1
            )
        return {
            "total_images": len(self.phase8_image_db),
            "image_types": type_counts,
            "dedup_threshold": self._threshold,
            "vector_backend": self._db.backend_name,
        }

    # ─── Internal helpers ─────────────────────────────────────────────────

    def _ocr_caption(self, image: Image.Image) -> str:
        """
        Extract text from image via Tesseract OCR.
        Returns empty string if Tesseract not installed.
        """
        try:
            import pytesseract
            if image.mode != "RGB":
                image = image.convert("RGB")
            text = pytesseract.image_to_string(image).strip()
            return text
        except ImportError:
            logger.warning(
                "pytesseract not installed — OCR skipped. "
                "Install: apt-get install tesseract-ocr && "
                "pip install pytesseract"
            )
            return ""
        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def _compute_embedding(
        self,
        image: Image.Image,
        caption: str,
    ) -> np.ndarray:
        """
        Compute image embedding.

        Strategy:
          1. Attempt BLIP-2 image embedding (GPU, ~8GB VRAM).
          2. If BLIP-2 unavailable or fails, fall back to
             SPECTER2 caption embedding.
          This ensures the pipeline runs on CPU during development
          without BLIP-2 loaded.
        """
        try:
            emb = self._embedder.embed_image(image)
            # BLIP-2 returns zeros on load failure — detect and fallback
            if np.allclose(emb, 0):
                raise ValueError("BLIP-2 returned zero embedding.")
            return emb
        except Exception as e:
            logger.debug(
                f"BLIP-2 embedding unavailable ({e}), "
                "falling back to SPECTER2 caption embedding."
            )
            if caption and caption.strip():
                return self._embedder.embed_caption(caption)
            # Last resort: random unit vector (marks as unique)
            logger.warning(
                "No caption available for fallback embedding. "
                "Using random unit vector — image will not deduplicate."
            )
            vec = np.random.randn(768).astype(np.float32)
            return vec / np.linalg.norm(vec)

    def _classify_image_type(self, caption_text: str) -> str:
        """
        Classify scientific image type from caption text.
        Keyword-based classification — fast, no model required.

        Returns one of IMAGE_TYPES.
        """
        if not caption_text:
            return "other"
        text = caption_text.lower()

        classification_rules = [
            ("microscopy",        ["microscop", "sem ", "tem ", "confocal",
                                   "fluorescen", "stain", "histolog"]),
            ("chemical_structure",["chemical", "molecule", "compound",
                                   "structure", "bond", "reaction",
                                   "synthesis", "nmr", "spectr"]),
            ("plot",              ["plot", "graph", "scatter", "regression",
                                   "correlation", "distribution", "histogram",
                                   "curve", "axis", "x-axis", "y-axis"]),
            ("chart",             ["chart", "bar chart", "pie chart",
                                   "table", "comparison"]),
            ("diagram",           ["diagram", "figure", "scheme",
                                   "schematic", "pathway", "circuit",
                                   "flow", "model", "illustration"]),
        ]

        for image_type, keywords in classification_rules:
            if any(kw in text for kw in keywords):
                return image_type

        return "other"
