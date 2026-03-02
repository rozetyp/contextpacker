"""
Simple in-memory cache for file indexes.
TTL-based expiration to handle repo updates and avoid memory bloat.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CacheEntry:
    """Cached file index data."""
    file_index: List[Dict]
    symbols_index: Dict
    tokens_raw_estimate: int
    created_at: float
    
    def is_expired(self, ttl_seconds: int = 300) -> bool:
        """Check if entry has expired."""
        return time.time() - self.created_at > ttl_seconds


class FileIndexCache:
    """
    In-memory cache for file indexes.
    
    Keyed by normalized repo URL with TTL-based expiration.
    Uses LRU-style eviction when at capacity.
    """
    
    def __init__(self, max_entries: int = 100, ttl_seconds: int = 300):
        """
        Initialize cache.
        
        Args:
            max_entries: Maximum number of cached repos
            ttl_seconds: Time-to-live for entries (default 5 min)
        """
        self._cache: Dict[str, CacheEntry] = {}
        self._max_entries = max_entries
        self._ttl = ttl_seconds
    
    def _normalize_url(self, repo_url: str) -> str:
        """Normalize URL for consistent cache keys."""
        return repo_url.rstrip("/").rstrip(".git").lower()
    
    def get(self, repo_url: str) -> Optional[CacheEntry]:
        """
        Get cached entry for repo URL.
        
        Args:
            repo_url: Repository URL
        
        Returns:
            CacheEntry if found and not expired, else None
        """
        key = self._normalize_url(repo_url)
        entry = self._cache.get(key)
        
        if entry and not entry.is_expired(self._ttl):
            return entry
        
        if entry:
            # Clean up expired entry
            del self._cache[key]
        
        return None
    
    def set(
        self,
        repo_url: str,
        file_index: List[Dict],
        symbols_index: Dict,
        tokens_raw_estimate: int,
    ) -> None:
        """
        Cache file index for repo URL.
        
        Args:
            repo_url: Repository URL
            file_index: List of file metadata
            symbols_index: AST-extracted symbols per file (keyed by path)
            tokens_raw_estimate: Estimated total tokens
        """
        # Evict oldest if at capacity
        if len(self._cache) >= self._max_entries:
            oldest_key = min(
                self._cache,
                key=lambda k: self._cache[k].created_at
            )
            del self._cache[oldest_key]
        
        key = self._normalize_url(repo_url)
        self._cache[key] = CacheEntry(
            file_index=file_index,
            symbols_index=symbols_index,
            tokens_raw_estimate=tokens_raw_estimate,
            created_at=time.time(),
        )
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        now = time.time()
        active = sum(
            1 for e in self._cache.values()
            if not e.is_expired(self._ttl)
        )
        return {
            "total_entries": len(self._cache),
            "active_entries": active,
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl,
        }


# Global cache instance
file_index_cache = FileIndexCache()
