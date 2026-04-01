"""
Logiik shared utility functions.
Canonical implementations — all modules import from here.
Do NOT reimplement these inline in other modules.
"""
import numpy as np
from scipy.spatial.distance import cosine
from typing import List
from logiik.utils.logging import log_event


def is_duplicate(
    new_embedding: np.ndarray,
    existing_embeddings: List[np.ndarray],
    threshold: float = 0.90
) -> bool:
    """
    Returns True if new_embedding is semantically duplicate of any
    embedding in existing_embeddings (cosine similarity > threshold).

    Args:
        new_embedding:       1-D numpy array of shape (dim,)
        existing_embeddings: List of 1-D numpy arrays
        threshold:           Similarity threshold (0–1). Default 0.90.

    Returns:
        bool: True if duplicate detected.
    """
    for emb in existing_embeddings:
        similarity = 1 - cosine(new_embedding, emb)
        if similarity > threshold:
            log_event("helpers", f"Duplicate detected (similarity={similarity:.4f})", "debug")
            return True
    return False


def validate_answer(answer_text: str, min_length: int = 30) -> bool:
    """
    Basic validation for a student model output.
    Returns True if answer passes minimum content checks.

    Args:
        answer_text: Raw string output from student model.
        min_length:  Minimum character length to be considered valid.

    Returns:
        bool: True if answer is non-empty and meets minimum length.
    """
    if not answer_text:
        return False
    stripped = answer_text.strip()
    return len(stripped) >= min_length


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks of fixed character length.

    Args:
        text:       Input text string.
        chunk_size: Maximum characters per chunk.
        overlap:    Number of characters to overlap between chunks.

    Returns:
        List of text chunk strings.
    """
    if not text or chunk_size <= 0:
        return []
    step = max(1, chunk_size - overlap)
    return [text[i:i + chunk_size] for i in range(0, len(text), step)]


def compute_saturation(
    new_embedding: np.ndarray,
    past_embeddings: List[np.ndarray],
    top_k: int = 5
) -> float:
    """
    Computes semantic saturation: average cosine similarity of new_embedding
    to the last top_k embeddings in past_embeddings.

    High saturation (approaching 1.0) indicates the model is producing
    semantically redundant outputs — phase may be complete.

    Args:
        new_embedding:   1-D numpy array.
        past_embeddings: List of prior embeddings (order preserved).
        top_k:           Number of recent embeddings to compare against.

    Returns:
        float: Mean similarity score in [0, 1]. Returns 0.0 if no history.
    """
    if not past_embeddings:
        return 0.0
    recent = past_embeddings[-top_k:]
    similarities = [1 - cosine(new_embedding, e) for e in recent]
    return float(np.mean(similarities))
