"""
Persistent repo content cache.

Caches cloned repositories to avoid re-cloning on repeat queries.
Uses LRU eviction when cache exceeds max size.
"""

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List, Dict

from .config import settings


@dataclass
class RepoCacheEntry:
    """Metadata for a cached repo."""
    repo_url: str
    repo_hash: str
    cached_at: float
    last_accessed: float
    size_bytes: int
    file_count: int
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RepoCacheEntry":
        return cls(**data)


class RepoCache:
    """
    Persistent cache for cloned repositories.
    
    Stores repos in ~/.contextpacker/repos/ with LRU eviction.
    
    Cache structure:
        ~/.contextpacker/
        ├── repos/
        │   ├── {repo_hash}/          # Cached repo contents
        │   │   ├── .metadata.json    # Cache metadata
        │   │   └── ...files...
        │   └── ...
        └── cache_index.json          # Global index for LRU
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        max_size_gb: float = 10.0,
    ):
        """
        Initialize repo cache.
        
        Args:
            cache_dir: Cache directory (default: ~/.contextpacker)
            max_size_gb: Maximum cache size in GB (default: 10GB)
        """
        self.cache_dir = cache_dir or Path.home() / ".contextpacker"
        self.repos_dir = self.cache_dir / "repos"
        self.index_path = self.cache_dir / "cache_index.json"
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        # Ensure directories exist
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize index
        self._index: Dict[str, RepoCacheEntry] = self._load_index()
    
    def _hash_url(self, repo_url: str) -> str:
        """Generate stable hash for repo URL."""
        normalized = repo_url.rstrip("/").rstrip(".git").lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _load_index(self) -> Dict[str, RepoCacheEntry]:
        """Load cache index from disk."""
        if not self.index_path.exists():
            return {}
        
        try:
            with open(self.index_path, "r") as f:
                data = json.load(f)
            return {
                k: RepoCacheEntry.from_dict(v)
                for k, v in data.items()
            }
        except Exception:
            return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        data = {k: v.to_dict() for k, v in self._index.items()}
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = Path(dirpath) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
        return total
    
    def _get_total_cache_size(self) -> int:
        """Get total size of all cached repos."""
        return sum(e.size_bytes for e in self._index.values())
    
    def _evict_lru(self, needed_bytes: int = 0) -> None:
        """
        Evict least-recently-used entries to free space.
        
        Args:
            needed_bytes: Additional bytes needed (evict until we have room)
        """
        target_size = self.max_size_bytes - needed_bytes
        current_size = self._get_total_cache_size()
        
        if current_size <= target_size:
            return
        
        # Sort by last_accessed (oldest first)
        sorted_entries = sorted(
            self._index.items(),
            key=lambda x: x[1].last_accessed
        )
        
        for repo_hash, entry in sorted_entries:
            if current_size <= target_size:
                break
            
            # Remove from disk
            repo_path = self.repos_dir / repo_hash
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            
            # Update tracking
            current_size -= entry.size_bytes
            del self._index[repo_hash]
        
        self._save_index()
    
    def get(self, repo_url: str) -> Optional[Path]:
        """
        Get cached repo path if available.
        
        Args:
            repo_url: Repository URL
        
        Returns:
            Path to cached repo directory, or None if not cached
        """
        repo_hash = self._hash_url(repo_url)
        entry = self._index.get(repo_hash)
        
        if entry is None:
            return None
        
        repo_path = self.repos_dir / repo_hash
        
        if not repo_path.exists():
            # Cache is stale, remove entry
            del self._index[repo_hash]
            self._save_index()
            return None
        
        # Update last accessed time
        entry.last_accessed = time.time()
        self._save_index()
        
        return repo_path
    
    def put(self, repo_url: str, source_path: Path) -> Path:
        """
        Add repo to cache by moving from source.
        
        Args:
            repo_url: Repository URL
            source_path: Path to cloned repo (will be moved, not copied)
        
        Returns:
            Path to cached repo location
        """
        repo_hash = self._hash_url(repo_url)
        dest_path = self.repos_dir / repo_hash
        
        # Calculate size before moving
        size_bytes = self._get_dir_size(source_path)
        file_count = sum(1 for _ in source_path.rglob("*") if _.is_file())
        
        # Evict if needed to make room
        self._evict_lru(needed_bytes=size_bytes)
        
        # Remove existing if present
        if dest_path.exists():
            shutil.rmtree(dest_path, ignore_errors=True)
        
        # Move source to cache
        shutil.move(str(source_path), str(dest_path))
        
        # Update index
        now = time.time()
        self._index[repo_hash] = RepoCacheEntry(
            repo_url=repo_url,
            repo_hash=repo_hash,
            cached_at=now,
            last_accessed=now,
            size_bytes=size_bytes,
            file_count=file_count,
        )
        self._save_index()
        
        return dest_path
    
    def invalidate(self, repo_url: str) -> bool:
        """
        Remove repo from cache.
        
        Args:
            repo_url: Repository URL
        
        Returns:
            True if repo was cached, False otherwise
        """
        repo_hash = self._hash_url(repo_url)
        entry = self._index.get(repo_hash)
        
        if entry is None:
            return False
        
        # Remove from disk
        repo_path = self.repos_dir / repo_hash
        if repo_path.exists():
            shutil.rmtree(repo_path, ignore_errors=True)
        
        # Remove from index
        del self._index[repo_hash]
        self._save_index()
        
        return True
    
    def clear(self) -> None:
        """Clear entire cache."""
        for repo_hash in list(self._index.keys()):
            repo_path = self.repos_dir / repo_hash
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
        
        self._index.clear()
        self._save_index()
    
    def stats(self) -> Dict:
        """Get cache statistics."""
        total_size = self._get_total_cache_size()
        return {
            "repos_cached": len(self._index),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "max_size_gb": self.max_size_bytes / (1024 * 1024 * 1024),
            "utilization_pct": round(100 * total_size / self.max_size_bytes, 1) if self.max_size_bytes > 0 else 0,
            "cache_dir": str(self.cache_dir),
        }


# Global instance with configurable max size
repo_cache = RepoCache(
    max_size_gb=getattr(settings, "REPO_CACHE_MAX_SIZE_GB", 10.0),
)
