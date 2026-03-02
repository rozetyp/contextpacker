"""
Unit tests for cache module (cache.py).

Tests in-memory caching behavior, TTL expiration, and LRU eviction.
"""

import pytest
import time


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_fresh_entry_not_expired(self):
        """Fresh entries should not be expired."""
        from context_packer.cache import CacheEntry
        
        entry = CacheEntry(
            file_index=[{"path": "test.py"}],
            symbols_index={},
            tokens_raw_estimate=100,
            created_at=time.time()
        )
        
        assert not entry.is_expired(ttl_seconds=300)
    
    def test_old_entry_is_expired(self):
        """Old entries should be marked as expired."""
        from context_packer.cache import CacheEntry
        
        # Entry created 10 minutes ago
        entry = CacheEntry(
            file_index=[],
            symbols_index={},
            tokens_raw_estimate=0,
            created_at=time.time() - 600
        )
        
        # Should be expired with 5 minute TTL
        assert entry.is_expired(ttl_seconds=300)
    
    def test_ttl_boundary(self):
        """Entry past TTL should be expired."""
        from context_packer.cache import CacheEntry
        
        # Entry created 301 seconds ago (just past TTL)
        entry = CacheEntry(
            file_index=[],
            symbols_index={},
            tokens_raw_estimate=0,
            created_at=time.time() - 301
        )
        
        # Should be expired with 5 minute (300s) TTL
        assert entry.is_expired(ttl_seconds=300)


class TestFileIndexCache:
    """Tests for FileIndexCache class."""
    
    def test_set_and_get(self):
        """Should store and retrieve entries."""
        from context_packer.cache import FileIndexCache
        
        cache = FileIndexCache(max_entries=10, ttl_seconds=300)
        
        cache.set(
            repo_url="https://github.com/test/repo",
            file_index=[{"path": "main.py"}],
            symbols_index={"main.py": {"symbols": ["main"], "doc": None}},
            tokens_raw_estimate=1000
        )
        
        entry = cache.get("https://github.com/test/repo")
        
        assert entry is not None
        assert entry.file_index == [{"path": "main.py"}]
        assert entry.symbols_index == {"main.py": {"symbols": ["main"], "doc": None}}
        assert entry.tokens_raw_estimate == 1000
    
    def test_get_nonexistent_returns_none(self):
        """Should return None for missing keys."""
        from context_packer.cache import FileIndexCache
        
        cache = FileIndexCache()
        
        result = cache.get("https://github.com/nonexistent/repo")
        
        assert result is None
    
    def test_url_normalization(self):
        """Should normalize URLs for consistent caching."""
        from context_packer.cache import FileIndexCache
        
        cache = FileIndexCache()
        
        # Set with one format
        cache.set(
            repo_url="https://github.com/test/repo.git",
            file_index=[{"path": "main.py"}],
            symbols_index={},
            tokens_raw_estimate=100
        )
        
        # Get with different formats should find the same entry
        assert cache.get("https://github.com/test/repo") is not None
        assert cache.get("https://github.com/test/repo/") is not None
        assert cache.get("https://github.com/test/REPO") is not None  # case insensitive
    
    def test_expired_entries_removed(self):
        """Expired entries should be removed on access."""
        from context_packer.cache import FileIndexCache
        
        cache = FileIndexCache(max_entries=10, ttl_seconds=1)  # 1 second TTL
        
        cache.set(
            repo_url="https://github.com/test/repo",
            file_index=[{"path": "main.py"}],
            symbols_index={},
            tokens_raw_estimate=100
        )
        
        # Entry should exist immediately
        assert cache.get("https://github.com/test/repo") is not None
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Entry should be gone
        assert cache.get("https://github.com/test/repo") is None
    
    def test_lru_eviction(self):
        """Should evict oldest entries when at capacity."""
        from context_packer.cache import FileIndexCache
        
        cache = FileIndexCache(max_entries=2, ttl_seconds=300)
        
        # Add 2 entries
        cache.set("https://github.com/test/repo1", [{"path": "1.py"}], {}, 100)
        time.sleep(0.01)  # Ensure different timestamps
        cache.set("https://github.com/test/repo2", [{"path": "2.py"}], {}, 100)
        
        # Both should exist
        assert cache.get("https://github.com/test/repo1") is not None
        assert cache.get("https://github.com/test/repo2") is not None
        
        # Add third entry - should evict oldest (repo1)
        cache.set("https://github.com/test/repo3", [{"path": "3.py"}], {}, 100)
        
        assert cache.get("https://github.com/test/repo1") is None  # Evicted
        assert cache.get("https://github.com/test/repo2") is not None
        assert cache.get("https://github.com/test/repo3") is not None
    
    def test_clear(self):
        """Should clear all entries."""
        from context_packer.cache import FileIndexCache
        
        cache = FileIndexCache()
        
        cache.set("https://github.com/test/repo1", [], {}, 0)
        cache.set("https://github.com/test/repo2", [], {}, 0)
        
        cache.clear()
        
        assert cache.get("https://github.com/test/repo1") is None
        assert cache.get("https://github.com/test/repo2") is None
    
    def test_stats(self):
        """Should return cache statistics."""
        from context_packer.cache import FileIndexCache
        
        cache = FileIndexCache(max_entries=10, ttl_seconds=300)
        
        cache.set("https://github.com/test/repo1", [], {}, 0)
        cache.set("https://github.com/test/repo2", [], {}, 0)
        
        stats = cache.stats()
        
        assert stats["total_entries"] == 2
        assert stats["active_entries"] == 2
        assert stats["max_entries"] == 10
        assert stats["ttl_seconds"] == 300


class TestGlobalCacheInstance:
    """Tests for the global file_index_cache singleton."""
    
    def test_global_instance_exists(self):
        """Should have a module-level cache instance."""
        from context_packer.cache import file_index_cache
        
        assert file_index_cache is not None
    
    def test_global_instance_is_usable(self, clean_cache):
        """Global cache should be functional."""
        from context_packer.cache import file_index_cache
        
        file_index_cache.set(
            repo_url="https://github.com/global/test",
            file_index=[{"path": "test.py"}],
            symbols_index={},
            tokens_raw_estimate=50
        )
        
        entry = file_index_cache.get("https://github.com/global/test")
        assert entry is not None
