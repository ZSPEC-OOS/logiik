"""
Logiik Phase 9 — PDF / Textbook Ingestion.

Ingests text and images from PDFs into Pinecone + Firebase.
Merges the simplified (lite) and full-scale ingestion specs
into a single pipeline controlled by a mode flag.

Modes:
  lite_mode  — <500 PDFs. Local paths used directly. No cloud
               staging. Reduced chunk size and batch size.
               Recommended for development and small datasets.
  full_mode  — 500–50,000 PDFs. Temporary cloud copy uploaded
               to Firebase Storage, deleted after ingestion.
               Larger chunks, higher GPU batch size.

Pipeline per PDF:
  1. [full_mode only] Upload temp copy to Firebase Storage.
  2. Extract text pages via PyMuPDF.
  3. Extract images via PyMuPDF + OCR via Tesseract.
  4. Pass images to Phase8ImagePipeline for dedup + storage.
  5. Chunk text into overlapping windows.
  6. Compute SPECTER2 embeddings (GPU-accelerated).
  7. Deduplicate chunks against Pinecone.
  8. Store new chunks: embedding → Pinecone, text → Firebase.
  9. Optionally cache hot chunks in Redis.
  10. [full_mode only] Delete temp cloud copy.
  11. Log ingestion metrics.

Usage:
    from logiik.ingestion.phase9_pdfs import Phase9PDFPipeline
    pipeline = Phase9PDFPipeline(mode='lite_mode')
    stats = pipeline.ingest_folder('path/to/pdfs/')
    print(stats)
"""
import os
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image
import numpy as np

from logiik.config import CONFIG
from logiik.embeddings.embed import get_embedder
from logiik.storage.vector_db import VectorDB
from logiik.storage.text_store import TextStore
from logiik.storage.cache import Cache
from logiik.ingestion.phase8_images import Phase8ImagePipeline
from logiik.utils.helpers import is_duplicate, chunk_text
from logiik.utils.logging import get_logger, log_event

logger = get_logger("ingestion.phase9_pdfs")

# ─── Mode configuration ───────────────────────────────────────────────────────

_INGESTION_CFG = CONFIG.get("ingestion", {})

INGESTION_CONFIG = {
    "lite_mode": {
        "chunk_size":  _INGESTION_CFG.get("lite_mode", {}).get("chunk_size",  512),
        "overlap":     _INGESTION_CFG.get("lite_mode", {}).get("overlap",     128),
        "batch_size":  _INGESTION_CFG.get("lite_mode", {}).get("batch_size",  5),
        "max_pdfs":    500,
        "cloud_stage": False,
    },
    "full_mode": {
        "chunk_size":  _INGESTION_CFG.get("full_mode", {}).get("chunk_size",  1024),
        "overlap":     _INGESTION_CFG.get("full_mode", {}).get("overlap",     256),
        "batch_size":  _INGESTION_CFG.get("full_mode", {}).get("batch_size",  20),
        "max_pdfs":    50000,
        "cloud_stage": True,
    },
}

DEDUP_THRESHOLD = _INGESTION_CFG.get("deduplication_threshold", 0.90)


# ─── Ingestion result ─────────────────────────────────────────────────────────

class IngestionResult:
    """Result record for a single PDF ingestion."""
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.chunks_total = 0
        self.chunks_new = 0
        self.chunks_duplicate = 0
        self.images_total = 0
        self.images_new = 0
        self.images_duplicate = 0
        self.duration_seconds = 0.0
        self.errors: List[str] = []

    def to_dict(self) -> Dict:
        return {
            "pdf_path": self.pdf_path,
            "chunks_total": self.chunks_total,
            "chunks_new": self.chunks_new,
            "chunks_duplicate": self.chunks_duplicate,
            "images_total": self.images_total,
            "images_new": self.images_new,
            "images_duplicate": self.images_duplicate,
            "duration_seconds": round(self.duration_seconds, 2),
            "errors": self.errors,
        }

    def __repr__(self):
        return (
            f"IngestionResult(pdf={Path(self.pdf_path).name}, "
            f"chunks={self.chunks_new}/{self.chunks_total}, "
            f"images={self.images_new}/{self.images_total}, "
            f"duration={self.duration_seconds:.1f}s)"
        )


# ─── Phase 9 Pipeline ─────────────────────────────────────────────────────────

class Phase9PDFPipeline:
    """
    Logiik Phase 9 PDF ingestion pipeline.

    Args:
        mode:           'lite_mode' or 'full_mode'.
                        Overrides config.yaml ingestion.simple_mode.
        vector_db:      VectorDB instance (created if not provided).
        text_store:     TextStore instance (created if not provided).
        cache:          Cache instance (created if not provided).
        phase8_pipeline: Phase8ImagePipeline instance. If not provided,
                        a new one is created. Pass an existing instance
                        to share the phase8_image_db across pipelines.
    """

    def __init__(
        self,
        mode: Optional[str] = None,
        vector_db: Optional[VectorDB] = None,
        text_store: Optional[TextStore] = None,
        cache: Optional[Cache] = None,
        phase8_pipeline: Optional[Phase8ImagePipeline] = None,
    ):
        # Resolve mode from arg → config → default
        if mode is None:
            simple = _INGESTION_CFG.get("simple_mode", True)
            mode = "lite_mode" if simple else "full_mode"
        if mode not in INGESTION_CONFIG:
            raise ValueError(
                f"Unknown mode: {mode!r}. "
                "Use 'lite_mode' or 'full_mode'."
            )

        self._mode = mode
        self._cfg = INGESTION_CONFIG[mode]
        self._db = vector_db or VectorDB()
        self._store = text_store or TextStore()
        self._cache = cache or Cache()
        self._embedder = get_embedder()
        self._phase8 = phase8_pipeline or Phase8ImagePipeline(
            vector_db=self._db,
            text_store=self._store,
            cache=self._cache,
        )

        # Embeddings seen in this session for fast local dedup
        # (supplements Pinecone query-based dedup)
        self._session_embeddings: List[np.ndarray] = []

        # Ingestion log for this session
        self._ingestion_log: List[IngestionResult] = []

        logger.info(
            f"Phase9PDFPipeline initialised: mode={mode}, "
            f"chunk_size={self._cfg['chunk_size']}, "
            f"overlap={self._cfg['overlap']}, "
            f"batch_size={self._cfg['batch_size']}, "
            f"cloud_stage={self._cfg['cloud_stage']}"
        )

        if mode == "lite_mode":
            log_event(
                "ingestion.phase9_pdfs",
                "Simple Mode active — recommended for 1–500 PDFs. "
                "For 500+ PDFs switch to full_mode.",
                level="info"
            )

    # ─── Public entry points ──────────────────────────────────────────────

    def ingest_folder(self, folder_path: str) -> Dict:
        """
        Ingest all PDFs in a folder.

        Args:
            folder_path: Path to folder containing PDF files.

        Returns:
            Aggregate stats dict across all PDFs.
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        pdf_files = sorted(folder.glob("*.pdf"))
        if not pdf_files:
            logger.warning(f"No PDF files found in {folder_path}")
            return {"pdfs_found": 0}

        max_pdfs = self._cfg["max_pdfs"]
        if len(pdf_files) > max_pdfs:
            logger.warning(
                f"{len(pdf_files)} PDFs found but mode={self._mode} "
                f"supports max {max_pdfs}. "
                f"Truncating to {max_pdfs}. "
                "Switch to full_mode for larger datasets."
            )
            pdf_files = pdf_files[:max_pdfs]

        logger.info(
            f"Phase 9: ingesting {len(pdf_files)} PDFs "
            f"from '{folder_path}' [{self._mode}]"
        )

        aggregate = {
            "pdfs_found": len(pdf_files),
            "pdfs_processed": 0,
            "pdfs_errored": 0,
            "chunks_total": 0,
            "chunks_new": 0,
            "chunks_duplicate": 0,
            "images_total": 0,
            "images_new": 0,
            "images_duplicate": 0,
            "total_duration_seconds": 0.0,
        }

        for pdf_path in pdf_files:
            result = self.ingest_pdf(str(pdf_path))
            aggregate["pdfs_processed"] += 1
            if result.errors:
                aggregate["pdfs_errored"] += 1
            for key in [
                "chunks_total", "chunks_new", "chunks_duplicate",
                "images_total", "images_new", "images_duplicate",
            ]:
                aggregate[key] += getattr(result, key)
            aggregate["total_duration_seconds"] += result.duration_seconds

        logger.info(f"Phase 9 folder ingestion complete: {aggregate}")
        return aggregate

    def ingest_pdf(self, pdf_path: str) -> IngestionResult:
        """
        Ingest a single PDF file.

        Args:
            pdf_path: Path to PDF file.

        Returns:
            IngestionResult with per-PDF statistics.
        """
        result = IngestionResult(pdf_path)
        start_time = time.time()
        temp_key = None

        try:
            logger.info(f"Phase 9: ingesting PDF: {Path(pdf_path).name}")

            # Step 1: Cloud staging (full_mode only)
            if self._cfg["cloud_stage"]:
                temp_key = self._upload_temp_cloud(pdf_path)

            # Step 2: Extract text and images
            text_pages, page_images = self._extract_pdf(pdf_path)

            # Step 3: Process images via Phase 8
            image_stats = self._phase8.process_pdf_images(
                pdf_path, page_images
            )
            result.images_total = image_stats["processed"]
            result.images_new = image_stats["new"]
            result.images_duplicate = image_stats["duplicates"]

            # Step 4: Chunk all text pages
            all_chunks: List[str] = []
            for page_text in text_pages:
                if page_text and page_text.strip():
                    chunks = chunk_text(
                        page_text,
                        self._cfg["chunk_size"],
                        self._cfg["overlap"]
                    )
                    all_chunks.extend(chunks)

            result.chunks_total = len(all_chunks)
            logger.debug(
                f"Extracted {len(all_chunks)} chunks "
                f"from {len(text_pages)} pages."
            )

            # Step 5: Embed + deduplicate + store chunks
            if all_chunks:
                new_count, dup_count = self._process_chunks(
                    all_chunks, pdf_path
                )
                result.chunks_new = new_count
                result.chunks_duplicate = dup_count

            # Step 6: Delete temp cloud copy (full_mode only)
            if temp_key:
                self._delete_temp_cloud(temp_key)
                temp_key = None

        except Exception as e:
            import traceback
            err_msg = f"{type(e).__name__}: {e}"
            result.errors.append(err_msg)
            logger.error(
                f"Phase 9 ingestion error for "
                f"{Path(pdf_path).name}: {err_msg}\n"
                f"{traceback.format_exc()}"
            )
            # Ensure temp cloud copy is cleaned up on error
            if temp_key:
                try:
                    self._delete_temp_cloud(temp_key)
                except Exception:
                    pass

        result.duration_seconds = time.time() - start_time
        self._ingestion_log.append(result)

        log_event(
            "ingestion.phase9_pdfs",
            f"PDF ingested: {result}",
            level="info"
        )
        return result

    def get_session_stats(self) -> Dict:
        """Return aggregate stats for this ingestion session."""
        if not self._ingestion_log:
            return {"pdfs_processed": 0}
        return {
            "mode": self._mode,
            "pdfs_processed": len(self._ingestion_log),
            "pdfs_errored": sum(
                1 for r in self._ingestion_log if r.errors
            ),
            "chunks_new": sum(r.chunks_new for r in self._ingestion_log),
            "chunks_duplicate": sum(
                r.chunks_duplicate for r in self._ingestion_log
            ),
            "images_new": sum(r.images_new for r in self._ingestion_log),
            "images_duplicate": sum(
                r.images_duplicate for r in self._ingestion_log
            ),
            "total_duration_seconds": round(
                sum(r.duration_seconds for r in self._ingestion_log), 2
            ),
            "phase8_stats": self._phase8.get_stats(),
            "vector_backend": self._db.backend_name,
        }

    # ─── Internal pipeline steps ──────────────────────────────────────────

    def _extract_pdf(
        self, pdf_path: str
    ) -> Tuple[List[str], List[Tuple[Image.Image, int]]]:
        """
        Extract text pages and images from a PDF via PyMuPDF.

        Returns:
            text_pages:  List of page text strings (one per page).
            page_images: List of (PIL.Image, page_number) tuples.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError(
                "PyMuPDF not installed. Run: pip install PyMuPDF"
            )

        doc = fitz.open(pdf_path)
        text_pages: List[str] = []
        page_images: List[Tuple[Image.Image, int]] = []

        for page_num, page in enumerate(doc):
            # Extract text
            text = page.get_text("text")
            text_pages.append(text)

            # Extract embedded images
            for img_info in page.get_images(full=True):
                try:
                    xref = img_info[0]
                    pix = fitz.Pixmap(doc, xref)

                    # Convert to RGB if necessary
                    if pix.n > 4:
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    img_pil = Image.frombytes(
                        "RGB",
                        [pix.width, pix.height],
                        pix.samples
                    )

                    # Skip very small images (likely icons/artifacts)
                    if pix.width >= 50 and pix.height >= 50:
                        page_images.append((img_pil, page_num))

                except Exception as e:
                    logger.debug(
                        f"Image extraction failed on page "
                        f"{page_num}: {e}"
                    )
                    continue

        doc.close()
        logger.debug(
            f"PDF extracted: {len(text_pages)} pages, "
            f"{len(page_images)} images — {Path(pdf_path).name}"
        )
        return text_pages, page_images

    def _process_chunks(
        self, chunks: List[str], source: str
    ) -> Tuple[int, int]:
        """
        Embed, deduplicate, and store text chunks.

        Deduplication strategy (two-level):
          1. Session-local: compare against embeddings seen this
             session (fast, in-memory).
          2. Pinecone query: for chunks passing local check,
             query Pinecone to catch cross-session duplicates.

        Returns:
            (new_count, duplicate_count)
        """
        new_count = 0
        dup_count = 0

        # Batch embed all chunks
        batch_size = self._cfg["batch_size"]
        all_embeddings = self._embedder.embed_texts(
            chunks, batch_size=batch_size
        )

        for chunk, emb in zip(chunks, all_embeddings):
            # Skip empty chunks
            if not chunk.strip():
                dup_count += 1
                continue

            # Level 1: session-local dedup
            if is_duplicate(emb, self._session_embeddings,
                           threshold=DEDUP_THRESHOLD):
                dup_count += 1
                continue

            # Level 2: Pinecone cross-session dedup
            if self._is_pinecone_duplicate(emb):
                dup_count += 1
                continue

            # New chunk — store
            chunk_id = str(uuid.uuid4())
            metadata = {
                "source": source,
                "text": chunk[:500],  # Pinecone metadata limit safety
                "phase": "phase9",
            }

            # Upsert to Pinecone
            self._db.upsert(
                id=f"phase9_{chunk_id}",
                embedding=emb,
                metadata=metadata
            )

            # Store full text in Firebase
            self._store.store_chunk(chunk_id, chunk, {
                "source": source,
                "phase": "phase9",
            })

            # Cache if enabled
            self._cache.set(chunk_id, chunk)

            # Add to session embeddings
            self._session_embeddings.append(emb)

            new_count += 1

        return new_count, dup_count

    def _is_pinecone_duplicate(self, emb: np.ndarray) -> bool:
        """
        Query Pinecone for nearest neighbor.
        Returns True if top match score exceeds dedup threshold.
        """
        try:
            matches = self._db.query(
                emb, top_k=1,
                filter={"phase": {"$eq": "phase9"}}
            )
            if matches and matches[0].score >= DEDUP_THRESHOLD:
                return True
            return False
        except Exception as e:
            # On query failure, treat as non-duplicate to avoid data loss
            logger.warning(f"Pinecone dedup query failed: {e}")
            return False

    def _upload_temp_cloud(self, pdf_path: str) -> str:
        """
        [full_mode only] Upload PDF to Firebase Storage as temp copy.
        Returns storage key for later deletion.

        NOTE: Firebase Storage REST upload requires multipart form.
        This implementation uses the Firebase Storage REST API.
        Requires FIREBASE_PROJECT set in environment.
        """
        import requests
        project = os.environ.get("FIREBASE_PROJECT", "nero-85ed0")
        api_key = os.environ.get("FIREBASE_API_KEY", "")
        bucket = f"{project}.appspot.com"
        filename = Path(pdf_path).name
        key = f"logiik_temp/{uuid.uuid4()}_{filename}"

        upload_url = (
            f"https://firebasestorage.googleapis.com/v0/b/"
            f"{bucket}/o?uploadType=media"
            f"&name={key}&key={api_key}"
        )

        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()

            r = requests.post(
                upload_url,
                headers={"Content-Type": "application/pdf"},
                data=pdf_bytes,
                timeout=120
            )
            if r.status_code in (200, 201):
                logger.debug(f"Temp cloud upload: key={key}")
                return key
            else:
                logger.warning(
                    f"Temp cloud upload failed "
                    f"(status={r.status_code}). "
                    "Proceeding with local file."
                )
                return ""
        except Exception as e:
            logger.warning(
                f"Temp cloud upload error: {e}. "
                "Proceeding with local file."
            )
            return ""

    def _delete_temp_cloud(self, key: str):
        """
        [full_mode only] Delete temporary cloud copy after ingestion.
        """
        if not key:
            return
        import requests
        project = os.environ.get("FIREBASE_PROJECT", "nero-85ed0")
        api_key = os.environ.get("FIREBASE_API_KEY", "")
        bucket = f"{project}.appspot.com"
        encoded_key = key.replace("/", "%2F")

        delete_url = (
            f"https://firebasestorage.googleapis.com/v0/b/"
            f"{bucket}/o/{encoded_key}?key={api_key}"
        )
        try:
            r = requests.delete(delete_url, timeout=30)
            if r.status_code in (200, 204):
                logger.debug(f"Temp cloud copy deleted: key={key}")
            else:
                logger.warning(
                    f"Temp cloud delete returned "
                    f"status={r.status_code} for key={key}"
                )
        except Exception as e:
            logger.warning(f"Temp cloud delete failed: {e}")
