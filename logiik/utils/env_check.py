"""
Logiik environment verification script.
Run before any training or inference to confirm all services
and hardware are correctly configured.

Usage:
    python -m logiik.utils.env_check
"""
import os
import sys
from pathlib import Path
from logiik.utils.logging import get_logger

logger = get_logger("env_check")


def check_python_version():
    """Verify Python >= 3.10."""
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    status = "PASS" if ok else "FAIL"
    logger.info(f"[{status}] Python version: {major}.{minor} (required: 3.10+)")
    return ok


def check_gpu():
    """
    Verify CUDA availability and report GPU info.
    Non-blocking — GPU not required for development phase.
    """
    try:
        import torch
        available = torch.cuda.is_available()
        if available:
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"[PASS] GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")

            # BLIP-2 in 8-bit requires ~8GB minimum
            if vram_gb < 8.0:
                logger.warning(
                    f"[WARN] VRAM {vram_gb:.1f}GB < 8GB minimum for BLIP-2 8-bit. "
                    "Image analysis (Phase 8) will be CPU-only."
                )
        else:
            logger.info(
                "[INFO] No GPU detected — running on CPU. "
                "GPU required for full training. Development mode active."
            )
        return True  # non-blocking
    except ImportError:
        logger.error("[FAIL] torch not installed. Run: pip install torch")
        return False


def check_pinecone():
    """Verify Pinecone credentials and index connectivity."""
    api_key = os.environ.get("PINECONE_API_KEY")
    host = os.environ.get("PINECONE_HOST")
    index_name = os.environ.get("PINECONE_INDEX")

    if not api_key or api_key.startswith("your_"):
        logger.warning("[WARN] PINECONE_API_KEY not set in environment.")
        return False

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=api_key)
        index = pc.Index(host=host)
        stats = index.describe_index_stats()
        dim = stats.get("dimension", "unknown")
        vector_count = stats.get("total_vector_count", 0)
        logger.info(
            f"[PASS] Pinecone connected: index='{index_name}', "
            f"dim={dim}, vectors={vector_count}"
        )
        if dim != 768:
            logger.error(
                f"[FAIL] Pinecone index dimension={dim}, expected 768. "
                "Delete and recreate index at dim=768."
            )
            return False
        return True
    except Exception as e:
        logger.error(f"[FAIL] Pinecone connection failed: {e}")
        return False


def check_firebase():
    """Verify Firebase REST API connectivity."""
    project = os.environ.get("FIREBASE_PROJECT", "nero-85ed0")
    api_key = os.environ.get("FIREBASE_API_KEY")

    if not api_key or api_key.startswith("your_"):
        logger.warning("[WARN] FIREBASE_API_KEY not set in environment.")
        return False

    try:
        import requests
        url = (
            f"https://firestore.googleapis.com/v1/projects/{project}"
            f"/databases/(default)/documents/logiik?key={api_key}"
        )
        r = requests.get(url, timeout=10)
        # 200 = exists, 404 = project exists but collection empty (both valid)
        ok = r.status_code in (200, 404)
        status = "PASS" if ok else "FAIL"
        logger.info(
            f"[{status}] Firebase REST API: project='{project}', "
            f"status_code={r.status_code}"
        )
        return ok
    except Exception as e:
        logger.error(f"[FAIL] Firebase connection failed: {e}")
        return False


def check_redis():
    """Check Redis — informational only, disabled by default."""
    from logiik.config import CONFIG
    if not CONFIG.get("cache", {}).get("enabled", False):
        logger.info("[SKIP] Redis cache disabled in config (cache.enabled=false)")
        return True  # non-blocking

    try:
        import redis
        host = os.environ.get("REDIS_HOST", "localhost")
        password = os.environ.get("REDIS_PASSWORD")
        r = redis.Redis(host=host, port=6379, password=password, socket_timeout=5)
        r.ping()
        logger.info(f"[PASS] Redis connected: host={host}")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Redis connection failed: {e}")
        return False


def check_embedding_model():
    """
    Verify SPECTER2 can be loaded.
    Downloads model on first run (~500MB) — skip in CI environments.
    """
    skip = os.environ.get("LOGIIK_SKIP_MODEL_CHECK", "false").lower() == "true"
    if skip:
        logger.info("[SKIP] Embedding model check skipped (LOGIIK_SKIP_MODEL_CHECK=true)")
        return True

    try:
        from sentence_transformers import SentenceTransformer
        logger.info("[INFO] Loading allenai/specter2_base — first run downloads ~500MB...")
        model = SentenceTransformer("allenai/specter2_base")
        test_emb = model.encode("enzyme kinetics at low pH")
        assert len(test_emb) == 768, f"Expected dim=768, got {len(test_emb)}"
        logger.info(f"[PASS] SPECTER2 loaded: output_dim={len(test_emb)}")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Embedding model check failed: {e}")
        return False


def check_env_file():
    """Verify .env file exists and is populated."""
    env_path = Path(".env")
    if not env_path.exists():
        logger.warning(
            "[WARN] .env file not found. "
            "Copy .env.example to .env and fill in credentials."
        )
        return False

    with open(env_path) as f:
        content = f.read()

    placeholders = [line for line in content.splitlines()
                    if "your_" in line and not line.strip().startswith("#")]
    if placeholders:
        logger.warning(
            f"[WARN] .env has {len(placeholders)} unfilled placeholder(s). "
            "Replace all 'your_...' values with real credentials."
        )
    else:
        logger.info("[PASS] .env file found and populated.")

    return True  # non-blocking


def run_all_checks() -> bool:
    """
    Run all environment checks.
    Returns True only if all blocking checks pass.
    Non-blocking checks (GPU, Redis, model download)
    log warnings but do not fail.
    """
    logger.info("=" * 60)
    logger.info("Logiik Environment Check")
    logger.info("=" * 60)

    # Load .env if present
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    results = {
        "python":    check_python_version(),   # blocking
        "env_file":  check_env_file(),          # non-blocking
        "gpu":       check_gpu(),               # non-blocking
        "pinecone":  check_pinecone(),          # blocking
        "firebase":  check_firebase(),          # blocking
        "redis":     check_redis(),             # non-blocking
        "embedding": check_embedding_model(),   # non-blocking
    }

    logger.info("=" * 60)
    passed = sum(results.values())
    total = len(results)
    logger.info(f"Results: {passed}/{total} checks passed")

    # Only python + pinecone + firebase are hard blockers
    blocking = ["python", "pinecone", "firebase"]
    blocking_ok = all(results[k] for k in blocking)

    if blocking_ok:
        logger.info("STATUS: Ready to proceed.")
    else:
        failed = [k for k in blocking if not results[k]]
        logger.error(f"STATUS: Blocked by: {failed}")

    logger.info("=" * 60)
    return blocking_ok


if __name__ == "__main__":
    ok = run_all_checks()
    sys.exit(0 if ok else 1)
