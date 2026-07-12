import threading
import time
from typing import Any, Callable, Optional


class CacheService:
    """A small thread-safe in-memory cache with optional TTL and memoize helper."""

    def __init__(self):
        self._store = {}  # key -> (value, expire_ts or None)
        self._lock = threading.Lock()

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        expire = time.time() + ttl if ttl and ttl > 0 else None
        with self._lock:
            self._store[key] = (value, expire)

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            value, expire = entry
            if expire is not None and time.time() > expire:
                # expired
                del self._store[key]
                return None
            return value

    def delete(self, key: str) -> None:
        with self._lock:
            if key in self._store:
                del self._store[key]

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def memoize(self, key_fn: Callable[..., str], ttl: Optional[float] = None):
        """Decorator to cache function results using key_fn(*args, **kwargs)."""

        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                key = key_fn(*args, **kwargs)
                val = self.get(key)
                if val is not None:
                    return val
                result = func(*args, **kwargs)
                self.set(key, result, ttl=ttl)
                return result

            return wrapper

        return decorator
