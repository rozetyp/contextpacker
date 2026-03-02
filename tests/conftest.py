"""
Pytest configuration and shared fixtures for ContextPacker tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest


# =============================================================================
# Async Support
# =============================================================================

@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_python_file() -> str:
    """Sample Python file content for symbol extraction tests."""
    return '''"""
Main orchestrator for context packing pipeline.
"""

from typing import List, Dict

MAX_FILES = 100
DEFAULT_TIMEOUT = 30

class PackRequest:
    """Request model for pack creation."""
    def __init__(self, query: str):
        self.query = query

class PackResponse:
    """Response model with packed files."""
    pass

async def create_pack(request: PackRequest) -> PackResponse:
    """Create a context pack from a repository."""
    pass

def _private_helper():
    """Private helper function (should be skipped)."""
    pass

def validate_input(data: Dict) -> bool:
    """Validate incoming request data."""
    return True
'''


@pytest.fixture
def sample_js_file() -> str:
    """Sample JavaScript file content for symbol extraction tests."""
    return '''/**
 * Authentication utilities for the application.
 */

export const AUTH_TIMEOUT = 5000;

export function authenticateUser(token) {
  return validateToken(token);
}

export async function refreshToken(oldToken) {
  // Implementation
}

export class AuthService {
  constructor() {
    this.tokens = new Map();
  }
}

export interface AuthConfig {
  timeout: number;
  retries: number;
}

function _internalHelper() {
  // Private helper
}
'''


@pytest.fixture
def sample_file_index() -> List[Dict]:
    """Sample file index for testing priority sorting and selection."""
    return [
        {"path": "src/main.py", "size_bytes": 1500, "depth": 2, "ext": ".py"},
        {"path": "src/auth/login.py", "size_bytes": 2000, "depth": 3, "ext": ".py"},
        {"path": "src/utils/helpers.py", "size_bytes": 800, "depth": 3, "ext": ".py"},
        {"path": "lib/core.js", "size_bytes": 3000, "depth": 2, "ext": ".js"},
        {"path": "tests/test_main.py", "size_bytes": 1200, "depth": 2, "ext": ".py"},
        {"path": "docs/README.md", "size_bytes": 500, "depth": 2, "ext": ".md"},
        {"path": ".github/workflows/ci.yml", "size_bytes": 300, "depth": 3, "ext": ".yml"},
        {"path": "examples/demo.py", "size_bytes": 600, "depth": 2, "ext": ".py"},
        {"path": "index.js", "size_bytes": 400, "depth": 1, "ext": ".js"},
        {"path": "main.py", "size_bytes": 350, "depth": 1, "ext": ".py"},
        {"path": "packages/core/src/index.ts", "size_bytes": 1800, "depth": 4, "ext": ".ts"},
        {"path": "app/models/user.py", "size_bytes": 900, "depth": 3, "ext": ".py"},
    ]


@pytest.fixture
def temp_repo(tmp_path) -> Path:
    """Create a temporary directory structure mimicking a repository."""
    # Create directory structure
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "auth").mkdir()
    (tmp_path / "lib").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs").mkdir()
    
    # Create sample files
    (tmp_path / "main.py").write_text('"""Entry point."""\ndef main(): pass')
    (tmp_path / "src" / "app.py").write_text('"""App module."""\nclass App: pass')
    (tmp_path / "src" / "auth" / "login.py").write_text('"""Login logic."""\ndef login(): pass')
    (tmp_path / "lib" / "utils.js").write_text('export function helper() {}')
    (tmp_path / "tests" / "test_app.py").write_text('def test_app(): assert True')
    (tmp_path / "docs" / "README.md").write_text('# Documentation')
    (tmp_path / "README.md").write_text('# Project')
    
    return tmp_path


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_response():
    """Factory for creating mock LLM responses."""
    def _create_response(paths: List[str], reasons: Dict[str, str] = None):
        if reasons:
            return [{"path": p, "reason": reasons.get(p, "relevant")} for p in paths]
        return paths
    return _create_response


@pytest.fixture
def mock_selector(monkeypatch, mock_llm_response):
    """Mock the LLM selector to avoid API calls in tests."""
    async def mock_call_selector_llm(
        file_tree_text: str,
        query: str,
        max_files: int = 10,
        include_reasons: bool = True,
        file_paths: List[str] = None,
    ):
        # Default response: pick first few source files
        selected = ["src/main.py", "src/auth/login.py", "lib/core.js"][:max_files]
        reasons = {p: f"relevant to {query}" for p in selected}
        cost = 0.0002
        return selected, cost, 100, reasons
    
    monkeypatch.setattr(
        "context_packer.selector.call_selector_llm",
        mock_call_selector_llm
    )


# =============================================================================
# Environment Fixtures
# =============================================================================

@pytest.fixture
def env_with_api_keys(monkeypatch):
    """Set up environment with required API keys for integration tests."""
    monkeypatch.setenv("LLM_API_KEY", "test-key-for-testing")
    monkeypatch.setenv("API_AUTH_TOKEN", "test-auth-token")


@pytest.fixture
def clean_cache():
    """Ensure caches are cleared before and after tests."""
    from context_packer.cache import file_index_cache
    file_index_cache.clear()
    yield
    file_index_cache.clear()
