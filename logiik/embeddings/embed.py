"""
Logiik Canonical Embedding Module.

Two embedding models:
  1. SPECTER2 (allenai/specter2) — text embeddings, dim=768
     Optimised for scientific documents and retrieval.
  2. BLIP-2 (Salesforce/blip2-opt-2.7b) — image embeddings, dim=768
     Loaded in 8-bit quantisation to fit on consumer GPUs (~8GB VRAM).
     Falls back to CPU if GPU unavailable (slow but functional).

All other modules import from here exclusively.
Do NOT instantiate embedding models inline elsewhere.

Usage:
    from logiik.embeddings.embed import get_embedder
    embedder = get_embedder()
    text_emb = embedder.embed_text("enzyme kinetics at low pH")
    image_emb = embedder.embed_image(pil_image)
    batch_embs = embedder.embed_texts(["text1", "text2", ...])
"""
import numpy as np
from typing import List, Optional, Union
from pathlib import Path
from PIL import Image

from logiik.config import CONFIG
from logiik.utils.logging import get_logger

logger = get_logger("embeddings.embed")


class TextEmbedder:
    """
    SPECTER2 text embedder for scientific documents.
    Output dimension: 768.
    Singleton — instantiated once via get_embedder().
    """

    MODEL_NAME = "allenai/specter2_base"
    OUTPUT_DIM = 768

    def __init__(self):
        self._model = None
        self._device = self._resolve_device()
        logger.info(
            f"TextEmbedder initialised (model={self.MODEL_NAME}, "
            f"device={self._device}, dim={self.OUTPUT_DIM})"
        )

    def _resolve_device(self) -> str:
        cfg_device = CONFIG.get("embeddings", {}).get("device", "auto")
        if cfg_device != "auto":
            return cfg_device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _load(self):
        """Lazy load — model downloaded/loaded on first embed call."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(
                f"Loading {self.MODEL_NAME} onto {self._device}... "
                "(first run downloads ~500MB)"
            )
            self._model = SentenceTransformer(
                self.MODEL_NAME,
                device=self._device
            )
            logger.info(f"{self.MODEL_NAME} loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SPECTER2: {e}. "
                "Run: pip install sentence-transformers"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Input text (scientific document, chunk, caption, etc.)

        Returns:
            np.ndarray of shape (768,)
        """
        self._load()
        if not text or not text.strip():
            logger.warning("embed_text called with empty string — returning zeros.")
            return np.zeros(self.OUTPUT_DIM, dtype=np.float32)
        emb = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        return emb.astype(np.float32)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Batch embed a list of text strings.

        Args:
            texts:      List of input strings.
            batch_size: Override config batch size.

        Returns:
            np.ndarray of shape (N, 768)
        """
        self._load()
        if not texts:
            return np.zeros((0, self.OUTPUT_DIM), dtype=np.float32)

        bs = batch_size or CONFIG.get("embeddings", {}).get("batch_size", 16)

        # Filter empty strings but preserve positions
        clean = [t if t and t.strip() else " " for t in texts]

        embs = self._model.encode(
            clean,
            batch_size=bs,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 50,
            normalize_embeddings=True
        )
        return embs.astype(np.float32)


class ImageEmbedder:
    """
    BLIP-2 image embedder for scientific figures and diagrams.
    Loaded in 8-bit quantisation to reduce VRAM requirement to ~8GB.
    Falls back to CPU (float32) if GPU unavailable.
    Output dimension: 768 (projected from BLIP-2 query tokens).

    NOTE: BLIP-2 requires ~8GB VRAM in 8-bit mode.
    On CPU, inference is slow (~30–60s per image) but functional.
    """

    MODEL_NAME = "Salesforce/blip2-opt-2.7b"
    OUTPUT_DIM = 768

    def __init__(self):
        self._model = None
        self._processor = None
        self._projection = None
        self._device = self._resolve_device()
        self._use_8bit = self._device == "cuda"
        logger.info(
            f"ImageEmbedder initialised (model={self.MODEL_NAME}, "
            f"device={self._device}, 8bit={self._use_8bit})"
        )

    def _resolve_device(self) -> str:
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb >= 8.0:
                    return "cuda"
                else:
                    logger.warning(
                        f"GPU VRAM={vram_gb:.1f}GB < 8GB required for BLIP-2. "
                        "Falling back to CPU for image embeddings."
                    )
                    return "cpu"
            return "cpu"
        except ImportError:
            return "cpu"

    def _load(self):
        """Lazy load BLIP-2 — heavy model, only loaded when needed."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Blip2Processor, Blip2Model

            logger.info(
                f"Loading {self.MODEL_NAME}... "
                "(first run downloads ~5GB — this may take several minutes)"
            )

            self._processor = Blip2Processor.from_pretrained(self.MODEL_NAME)

            if self._use_8bit:
                # 8-bit quantisation via bitsandbytes (~8GB VRAM)
                self._model = Blip2Model.from_pretrained(
                    self.MODEL_NAME,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                # CPU fallback — float32
                self._model = Blip2Model.from_pretrained(
                    self.MODEL_NAME,
                    torch_dtype=torch.float32
                )
                self._model = self._model.to("cpu")

            self._model.eval()

            # Linear projection from BLIP-2 query token dim → 768
            # BLIP-2 query tokens: shape depends on model config
            # We mean-pool query tokens then project to OUTPUT_DIM
            blip2_hidden = self._model.config.qformer_config.hidden_size
            self._projection = torch.nn.Linear(blip2_hidden, self.OUTPUT_DIM)
            if self._device == "cuda":
                self._projection = self._projection.cuda()

            logger.info(
                f"{self.MODEL_NAME} loaded. "
                f"QFormer hidden_size={blip2_hidden}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to load BLIP-2: {e}. "
                "Ensure bitsandbytes and transformers>=4.40 are installed."
            )

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """
        Embed a single PIL image using BLIP-2 query tokens.

        Args:
            image: PIL.Image.Image (RGB recommended)

        Returns:
            np.ndarray of shape (768,)
        """
        self._load()
        try:
            import torch

            if image.mode != "RGB":
                image = image.convert("RGB")

            inputs = self._processor(
                images=image,
                return_tensors="pt"
            )

            if self._device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()
                         if hasattr(v, 'cuda')}

            with torch.no_grad():
                outputs = self._model.get_qformer_features(**inputs)
                # query_output shape: (1, num_query_tokens, hidden_size)
                query_tokens = outputs.last_hidden_state  # (1, N, H)
                # Mean-pool across query tokens → (1, H)
                pooled = query_tokens.mean(dim=1)
                # Project to OUTPUT_DIM → (1, 768)
                projected = self._projection(pooled)
                # L2-normalise
                projected = torch.nn.functional.normalize(projected, dim=-1)
                emb = projected.squeeze(0).cpu().float().numpy()

            return emb.astype(np.float32)

        except Exception as e:
            logger.error(f"Image embedding failed: {e}")
            return np.zeros(self.OUTPUT_DIM, dtype=np.float32)

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """
        Embed a list of PIL images.

        Returns:
            np.ndarray of shape (N, 768)
        """
        if not images:
            return np.zeros((0, self.OUTPUT_DIM), dtype=np.float32)
        embs = [self.embed_image(img) for img in images]
        return np.stack(embs).astype(np.float32)


class Embedder:
    """
    Unified embedder exposing both text (SPECTER2) and image (BLIP-2).
    Lazy loads each model only when first needed.

    This is the canonical entry point for all embedding operations.
    """

    def __init__(self):
        self._text = TextEmbedder()
        self._image = ImageEmbedder()

    def embed_text(self, text: str) -> np.ndarray:
        """Single text → (768,) embedding."""
        return self._text.embed_text(text)

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> np.ndarray:
        """Batch text list → (N, 768) embeddings."""
        return self._text.embed_texts(texts, batch_size)

    def embed_image(self, image: Image.Image) -> np.ndarray:
        """Single PIL image → (768,) embedding."""
        return self._image.embed_image(image)

    def embed_images(self, images: List[Image.Image]) -> np.ndarray:
        """Batch PIL images → (N, 768) embeddings."""
        return self._image.embed_images(images)

    def embed_caption(self, caption: str) -> np.ndarray:
        """
        Embed an image caption as text.
        Used when BLIP-2 is unavailable or for semantic deduplication
        of images via their OCR-extracted captions.
        """
        return self._text.embed_text(caption)


# ─── Module-level singleton ───────────────────────────────────────────────────
# Import this directly for convenience:
#   from logiik.embeddings.embed import get_embedder
#   embedder = get_embedder()

_embedder_instance: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """
    Returns the module-level Embedder singleton.
    Creates it on first call. Thread-safe for read-only use.
    """
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = Embedder()
    return _embedder_instance
