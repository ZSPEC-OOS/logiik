"""
Logiik Redis Cache — optional hot knowledge cache.

Disabled by default via config.yaml: cache.enabled = false.
All methods are safe to call when Redis is disabled — they
return None/False silently rather than raising exceptions.

To enable:
  1. Set up Redis Cloud (redis.com — free tier available)
  2. Add REDIS_HOST and REDIS_PASSWORD to .env
  3. Set cache.enabled: true in config.yaml
"""
import os
from typing import Optional

from logiik.config import CONFIG
from logiik.utils.logging import get_logger

logger = get_logger("storage.cache")


class Cache:
    """
    Redis-backed cache for frequently accessed knowledge entries.

    Usage:
        from logiik.storage.cache import Cache
        cache = Cache()
        cache.set("chunk_001", "enzyme text...")
        text = cache.get("chunk_001")  # returns None if disabled or missing
    """

    def __init__(self):
        self._enabled = CONFIG.get("cache", {}).get("enabled", False)
        self._client = None

        if self._enabled:
            self._init_redis()
        else:
            logger.info("Cache disabled (cache.enabled=false in config). "
                       "All cache calls will be no-ops.")

    def _init_redis(self):
        try:
            import redis
            host = os.environ.get("REDIS_HOST", "localhost")
            port = int(CONFIG.get("cache", {}).get("port", 6379))
            password = os.environ.get("REDIS_PASSWORD")
            self._client = redis.Redis(
                host=host, port=port,
                password=password,
                socket_timeout=5,
                decode_responses=True
            )
            self._client.ping()
            self._ttl = CONFIG.get("cache", {}).get("ttl_seconds", 3600)
            logger.info(f"Redis cache connected: host={host}")
        except Exception as e:
            logger.warning(
                f"Redis connection failed: {e}. "
                "Cache disabled — falling back to no-op mode."
            )
            self._enabled = False
            self._client = None

    def set(self, id: str, text: str) -> bool:
        """
        Store text in cache with TTL.
        Returns False silently if cache is disabled or unavailable.
        """
        if not self._enabled or not self._client:
            return False
        try:
            self._client.setex(f"logiik:knowledge:{id}", self._ttl, text)
            logger.debug(f"Cache SET: id={id}")
            return True
        except Exception as e:
            logger.warning(f"Cache SET failed for id={id}: {e}")
            return False

    def get(self, id: str) -> Optional[str]:
        """
        Retrieve text from cache.
        Returns None if disabled, missing, or on error.
        """
        if not self._enabled or not self._client:
            return None
        try:
            value = self._client.get(f"logiik:knowledge:{id}")
            if value:
                logger.debug(f"Cache HIT: id={id}")
            return value
        except Exception as e:
            logger.warning(f"Cache GET failed for id={id}: {e}")
            return None

    def delete(self, id: str) -> bool:
        """Remove a single entry from cache."""
        if not self._enabled or not self._client:
            return False
        try:
            self._client.delete(f"logiik:knowledge:{id}")
            return True
        except Exception as e:
            logger.warning(f"Cache DELETE failed for id={id}: {e}")
            return False

    def flush_all(self) -> bool:
        """
        Clear all Logiik cache entries.
        Only deletes keys with 'logiik:knowledge:' prefix.
        """
        if not self._enabled or not self._client:
            return False
        try:
            keys = self._client.keys("logiik:knowledge:*")
            if keys:
                self._client.delete(*keys)
                logger.info(f"Cache flushed: {len(keys)} entries removed")
            return True
        except Exception as e:
            logger.warning(f"Cache flush failed: {e}")
            return False

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._client is not None
