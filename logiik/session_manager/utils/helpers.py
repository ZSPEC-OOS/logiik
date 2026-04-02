"""
Logiik Session Manager Utilities.

Logging and GPU monitoring helpers specific to the
session manager. Wraps the canonical logiik logging
system with session-prefixed output.
"""
import subprocess
from typing import Dict, Any

from logiik.utils.logging import get_logger


class SessionLogger:
    """
    Thin wrapper around logiik canonical logger
    with session-specific prefix.

    Usage:
        logger = SessionLogger("session_manager")
        logger.info("Model loaded.")
        logger.warning("Low VRAM.")
        logger.error("Generation failed.")
    """

    def __init__(self, name: str):
        self._logger = get_logger(f"session_manager.{name}")

    def info(self, msg: str):
        self._logger.info(msg)

    def warning(self, msg: str):
        self._logger.warning(msg)

    def error(self, msg: str):
        self._logger.error(msg)

    def debug(self, msg: str):
        self._logger.debug(msg)


def get_gpu_snapshot() -> Dict[str, Any]:
    """
    Return current GPU memory and utilisation stats.
    Safe to call at any time — returns CPU-mode dict if no GPU.

    Returns:
        Dict with keys:
          gpu_available, device_name, vram_total_gb,
          vram_used_gb, vram_free_gb, utilisation_pct,
          temperature_c
    """
    snapshot = {
        "gpu_available": False,
        "device_name": "CPU",
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_free_gb": 0.0,
        "utilisation_pct": None,
        "temperature_c": None,
    }

    # PyTorch VRAM
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            total = props.total_memory / 1e9
            used = torch.cuda.memory_allocated(0) / 1e9
            snapshot.update({
                "gpu_available": True,
                "device_name": props.name,
                "vram_total_gb": round(total, 2),
                "vram_used_gb": round(used, 2),
                "vram_free_gb": round(total - used, 2),
            })
    except Exception:
        pass

    # nvidia-smi utilisation + temperature
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 2:
                snapshot["utilisation_pct"] = float(parts[0].strip())
                snapshot["temperature_c"] = float(parts[1].strip())
    except Exception:
        pass

    return snapshot


def format_uptime(seconds: float) -> str:
    """
    Format uptime seconds as human-readable string.

    Examples:
        45.0    → '45s'
        130.0   → '2m 10s'
        3661.0  → '1h 1m 1s'
    """
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"
