"""
Git utilities for cloning repositories and building file indexes.
Handles async clone, PAT injection, timeouts, and file tree generation.
"""

import asyncio
import contextlib
import re
from pathlib import Path
from typing import Dict, List, Optional

from .config import settings


# =============================================================================
# Path Priority Scoring
# =============================================================================

# Priority scores - extract to config if per-repo customization is needed
PRIORITY_HIGH_SOURCE = 10      # src/, lib/, packages/*/src/
PRIORITY_ROOT_ENTRY = 5        # Root-level source files (index.js, main.py)
PRIORITY_NEUTRAL = 0           # Default
PRIORITY_LOW_SIGNAL = -15      # test/, examples/, docs/, docs_src/ — strong penalty so they
                               # don't outrank real source files even when filename matches query
PRIORITY_INFRA = -10           # .github/, .circleci/, eslint-plugin-*

# High-priority directories (source code)
HIGH_PRIORITY_PATTERNS = [
    re.compile(r"^(?:packages/[^/]+/)?src/"),  # src/ or packages/*/src/
    re.compile(r"^(?:packages/[^/]+/)?lib/"),  # lib/ or packages/*/lib/
    re.compile(r"^core/"),                      # core/
    re.compile(r"^internal/"),                  # internal/
    re.compile(r"^cmd/"),                       # Go convention
    re.compile(r"^pkg/"),                       # Go convention
    re.compile(r"^app/"),                       # Rails, some Python
]

# Low-priority directories (infrastructure, tooling, docs)
LOW_PRIORITY_DIRS = {
    ".github", ".circleci", ".gitlab", ".travis",
    "docs", "doc", "documentation", "docs_src", "doc_src",
    "examples", "example", "samples", "sample", "demo", "demos",
    "test", "tests", "__tests__", "spec", "specs", "__mocks__",
    "fixtures", "fixture", "__fixtures__",
    "scripts", "script", "tools", "tool",
    "benchmarks", "benchmark", "perf",
    ".husky", ".changeset",
    # i18n data and DB migrations are data/infra, not source logic
    "locale", "locales", "i18n", "migrations", "migration",
}

# Low-priority path patterns (eslint plugins, configs, etc.)
LOW_PRIORITY_PATTERNS = [
    re.compile(r"eslint-plugin-"),
    re.compile(r"^\."),  # dotfiles at root
    re.compile(r"config/"),
    re.compile(r"configs/"),
]


def compute_path_priority(path: str, query_tokens: Optional[set] = None) -> int:
    """
    Compute priority score for a file path.
    Combines structural heuristics with query-aware lexical matching.
    """
    score = PRIORITY_NEUTRAL
    path_obj = Path(path)
    parts = path_obj.parts
    ext = path_obj.suffix.lower()
    
    # 1. Structural Heuristics
    SOURCE_CODE_EXTENSIONS = {'.js', '.ts', '.jsx', '.tsx', '.py', '.go', '.rs', 
                              '.java', '.rb', '.php', '.c', '.cpp', '.h', '.hpp',
                              '.cs', '.swift', '.kt', '.scala', '.clj', '.ex', '.exs'}
    if ext in SOURCE_CODE_EXTENSIONS:
        score += 3
    
    # High-priority patterns
    for pattern in HIGH_PRIORITY_PATTERNS:
        if pattern.search(path):
            score += PRIORITY_HIGH_SOURCE
            break
    
    # Root entry points
    if len(parts) == 1 and ext in SOURCE_CODE_EXTENSIONS:
        score += PRIORITY_ROOT_ENTRY
    
    # Low-priority directories
    for part in parts[:-1]:
        if part.lower() in LOW_PRIORITY_DIRS:
            score += PRIORITY_LOW_SIGNAL
            break
            
    # Check low-priority patterns
    for pattern in LOW_PRIORITY_PATTERNS:
        if pattern.search(path):
            score += PRIORITY_INFRA
            break

    # 2. Dynamic Query Match (Query-Aware Priority)
    # Only applied to source code files — docs/tutorials named after features
    # must not outrank the actual implementation files in the tree.
    if query_tokens and ext in SOURCE_CODE_EXTENSIONS:
        # Extract keywords from path parts
        path_name = path_obj.name.lower()
        path_dir = str(path_obj.parent).lower()
        
        # Check filename match (Highest Boost)
        name_words = set(re.findall(r"\w+", path_name))
        name_matches = name_words.intersection(query_tokens)
        if name_matches:
            score += (len(name_matches) * 25)
            
        # Check directory match (High Boost)
        # Avoid double counting if filename matches too
        dir_words = set(re.findall(r"\w+", path_dir)) - name_words
        dir_matches = dir_words.intersection(query_tokens)
        if dir_matches:
            score += (len(dir_matches) * 15)

    return score


def priority_sort_paths(file_index: List[Dict], query: Optional[str] = None) -> List[Dict]:
    """
    Sort file index by priority for better tree truncation.
    """
    query_tokens = set(re.findall(r"\w+", query.lower())) if query else None
    
    def sort_key(entry: Dict) -> tuple:
        path = entry["path"]
        priority = compute_path_priority(path, query_tokens)
        depth = entry.get("depth", len(Path(path).parts))
        return (-priority, depth, path)
    
    return sorted(file_index, key=sort_key)


# =============================================================================
# Constants
# =============================================================================

# Repo size limits
MAX_FILES_LIMIT = 50_000
MAX_TOTAL_SIZE_MB = 500

IGNORED_DIRS = {
    ".git", "node_modules", "dist", "build", "target", ".next", ".nuxt",
    "venv", "env", ".venv", "virtualenv", "vendor", "__pycache__", ".pycache",
    "coverage", ".coverage", ".idea", ".vscode", ".cache", "tmp", ".tmp",
    ".eggs", "*.egg-info", ".tox", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "htmlcov", ".sass-cache", "bower_components", ".gradle", ".mvn", "Pods",
    ".dart_tool", ".pub-cache",
}

IGNORED_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".o", ".a", ".dylib", ".dll", ".exe",
    ".bin", ".dat", ".db", ".sqlite", ".sqlite3",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg", ".webp", ".bmp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".webm",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".lock", ".min.js", ".min.css",
}


# =============================================================================
# Custom Exceptions for Private Repos
# =============================================================================

class AuthenticationError(Exception):
    """Raised when PAT authentication fails."""
    pass


class RepoNotFoundError(Exception):
    """Raised when repository doesn't exist or isn't accessible."""
    pass


class CloneTimeoutError(Exception):
    """Raised when git clone takes too long."""
    pass


class CloneFailedError(Exception):
    """Raised when git clone fails for other reasons (e.g. invalid URL)."""
    pass


# =============================================================================
# Async clone
# =============================================================================

async def clone_repo_async(
    repo_url: str,
    target_dir: Path,
    github_token: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> None:
    """
    Asynchronously clone a Git repository with PAT injection and timeout.
    """
    if timeout_s is None:
        timeout_s = settings.MAX_CLONE_TIMEOUT_S
    
    # Inject PAT into URL for GitHub repos (avoids rate limiting)
    final_url = repo_url
    if github_token and "github.com" in repo_url:
        clean_url = repo_url.replace("https://", "")
        final_url = f"https://oauth2:{github_token}@{clean_url}"
    
    # Ensure parent directory exists
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Set environment to disable prompts
    env = {
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "",
    }
    
    process = await asyncio.create_subprocess_exec(
        "git", "clone",
        "--depth", "1",
        "--single-branch",
        final_url,
        str(target_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**dict(__import__('os').environ), **env}
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_s
        )
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
            await process.wait()
        raise CloneTimeoutError("REPO_CLONE_TIMEOUT")
    
    # Check for clone failure
    if process.returncode != 0:
        error_msg = stderr.decode(errors="ignore").lower()
        
        # Detect missing/private repo
        if any(p in error_msg for p in ["not found", "404", "could not read username"]):
            raise RepoNotFoundError(f"Repository not found or private: {repo_url}")
            
        raise CloneFailedError(f"REPO_CLONE_FAILED: {error_msg[:200]}")


async def clone_private_repo(
    repo_url: str,
    token: str,
    target_dir: Path,
    branch: Optional[str] = None,
    timeout_s: Optional[float] = None,
) -> None:
    from urllib.parse import urlparse
    if timeout_s is None:
        timeout_s = settings.MAX_CLONE_TIMEOUT_S
    
    parsed = urlparse(repo_url)
    if parsed.netloc not in ("github.com",):
        raise ValueError(f"Unsupported host: {parsed.netloc}")
    
    path = parsed.path.rstrip('/')
    if path.endswith(".git"):
        path = path[:-4]
    auth_url = f"https://{token}@github.com{path}.git"
    
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    
    cmd = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["--branch", branch])
    else:
        cmd.append("--single-branch")
    cmd.extend([auth_url, str(target_dir)])
    
    env = {"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": ""}
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        env={**dict(__import__('os').environ), **env}
    )
    
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        with contextlib.suppress(ProcessLookupError):
            process.kill()
            await process.wait()
        raise RuntimeError("REPO_CLONE_TIMEOUT")
    
    if process.returncode != 0:
        error_msg = stderr.decode(errors="ignore").lower()
        if any(p in error_msg for p in ["authentication failed", "401", "403"]):
            raise AuthenticationError("Invalid token or insufficient permissions.")
        if any(p in error_msg for p in ["not found", "404"]):
            raise RepoNotFoundError(f"Repository not found: {repo_url}")
        
        safe_error = re.sub(r'(ghp_|github_pat_)[A-Za-z0-9_]+', '[REDACTED]', stderr.decode(errors="ignore")[:200])
        raise RuntimeError(f"REPO_CLONE_FAILED: {safe_error}")


# =============================================================================
# File indexing
# =============================================================================

def build_file_index(root_dir: Path) -> List[Dict]:
    items: List[Dict] = []
    max_depth = settings.MAX_TREE_DEPTH
    max_file_size = settings.MAX_FILE_SIZE_BYTES
    
    for file_path in root_dir.rglob("*"):
        if file_path.is_dir(): continue
        try: rel_path = file_path.relative_to(root_dir)
        except ValueError: continue
        
        rel_path_str = str(rel_path)
        depth = len(rel_path.parts)
        if depth > max_depth: continue
        if any(part in IGNORED_DIRS for part in rel_path.parts): continue
        if file_path.suffix.lower() in IGNORED_EXTENSIONS: continue
        
        try: size_bytes = file_path.stat().st_size
        except (OSError, IOError): continue
        if size_bytes > max_file_size or size_bytes == 0: continue
        
        try:
            with open(file_path, "rb") as f:
                if b"\x00" in f.read(1024): continue
        except (OSError, IOError): continue
        
        items.append({"path": rel_path_str, "size_bytes": size_bytes, "ext": file_path.suffix.lower(), "depth": depth})
    
    items.sort(key=lambda x: x["path"])
    return items


def validate_repo_size(file_index: List[Dict]) -> None:
    if len(file_index) > MAX_FILES_LIMIT:
        raise RuntimeError(f"REPO_TOO_LARGE: {len(file_index):,} files (limit: {MAX_FILES_LIMIT:,})")
    total_mb = sum(f.get("size_bytes", 0) for f in file_index) / (1024 * 1024)
    if total_mb > MAX_TOTAL_SIZE_MB:
        raise RuntimeError(f"REPO_TOO_LARGE: {total_mb:.0f}MB (limit: {MAX_TOTAL_SIZE_MB}MB)")


# =============================================================================
# Tree text generation
# =============================================================================

def file_index_to_tree_text(
    file_index: List[Dict],
    max_chars: Optional[int] = None,
    query: Optional[str] = None,
) -> str:
    if not file_index: return "(empty repository)"
    max_chars = max_chars or settings.MAX_TREE_CHARS
    query_tokens = set(re.findall(r"\w+", query.lower())) if query else set()
    
    scored_index = []
    for entry in file_index:
        entry["_score"] = compute_path_priority(entry["path"], query_tokens)
        scored_index.append(entry)
    scored_index.sort(key=lambda x: x["path"])
    
    lines, current_len, shown_dirs = [], 0, set()
    dir_to_children = {}
    for entry in scored_index:
        p = str(Path(entry["path"]).parent)
        if p not in dir_to_children: dir_to_children[p] = []
        dir_to_children[p].append(entry)

    MAX_SIBLINGS_PER_DIR = 30
    for entry in scored_index:
        path = entry["path"]
        parts = Path(path).parts
        
        for i in range(len(parts) - 1):
            dir_path = "/".join(parts[:i + 1])
            if dir_path not in shown_dirs:
                indent, line = "  " * i, f"{parts[i]}/"
                if current_len + len(line) + 1 < max_chars:
                    lines.append(line); current_len += len(line) + 1; shown_dirs.add(dir_path)
        
        parent_path = str(Path(path).parent)
        siblings = dir_to_children.get(parent_path, [])
        sib_idx = next((i for i, s in enumerate(siblings) if s["path"] == path), 0)
        
        if len(siblings) > MAX_SIBLINGS_PER_DIR and sib_idx >= MAX_SIBLINGS_PER_DIR:
            if sib_idx == MAX_SIBLINGS_PER_DIR:
                line = f"{('  ' * (len(parts) - 1))}... (+{len(siblings) - MAX_SIBLINGS_PER_DIR} more files)"
                if current_len + len(line) + 1 < max_chars:
                    lines.append(line); current_len += len(line) + 1
            continue

        line = f"{('  ' * (len(parts) - 1))}{parts[-1]}"
        if current_len + len(line) + 1 < max_chars:
            lines.append(line); current_len += len(line) + 1
        else:
            if not lines or "... (hard" not in lines[-1]: lines.append("... (tree exceeds token limit)")
            break
    return "\n".join(lines)


def file_index_to_enriched_tree(
    file_index: List[Dict],
    symbols_index: Dict[str, Dict],
    max_chars: Optional[int] = None,
    query: Optional[str] = None,
) -> str:
    from .symbols import format_symbols_hint
    if not file_index: return "(empty repository)"
    max_chars = max_chars or settings.MAX_TREE_CHARS
    query_tokens = set(re.findall(r"\w+", query.lower())) if query else set()
    
    scored_index = []
    for entry in file_index:
        entry["_score"] = compute_path_priority(entry["path"], query_tokens)
        scored_index.append(entry)
    # Sort by priority descending so high-value files survive tree truncation.
    # Tests, docs, migrations get cut first; source files appear first in the tree.
    scored_index.sort(key=lambda x: (-x["_score"], x["path"]))
    
    lines, current_len, shown_dirs = [], 0, set()
    dir_to_children = {}
    for entry in scored_index:
        p = str(Path(entry["path"]).parent)
        if p not in dir_to_children: dir_to_children[p] = []
        dir_to_children[p].append(entry)

    DETAIL_THRESHOLD, MAX_SIBLINGS_PER_DIR = 15, 30
    for entry in scored_index:
        path = entry["path"]
        parts = Path(path).parts
        score = entry["_score"]
        
        for i in range(len(parts) - 1):
            dir_path = "/".join(parts[:i + 1])
            if dir_path not in shown_dirs:
                indent, line = "  " * i, f"{parts[i]}/"
                full_line = f"{indent}{line}"
                if current_len + len(full_line) + 1 < max_chars:
                    lines.append(full_line); current_len += len(full_line) + 1; shown_dirs.add(dir_path)
        
        parent_path = str(Path(path).parent)
        siblings = dir_to_children.get(parent_path, [])
        sib_idx = next((i for i, s in enumerate(siblings) if s["path"] == path), 0)
        
        if len(siblings) > MAX_SIBLINGS_PER_DIR and sib_idx >= MAX_SIBLINGS_PER_DIR:
            if sib_idx == MAX_SIBLINGS_PER_DIR:
                line = f"{('  ' * (len(parts) - 1))}... (+{len(siblings) - MAX_SIBLINGS_PER_DIR} more files)"
                if current_len + len(line) + 1 < max_chars:
                    lines.append(line); current_len += len(line) + 1
            continue

        symbol_info = symbols_index.get(path)
        # Show symbols if: high score, query term literally in path, or path stem matches a query term
        # (e.g. path "config.py" stem "config" matches query token "configuration")
        path_name_stems = {w for w in re.findall(r"\w+", path.lower().rsplit(".", 1)[0]) if len(w) >= 4}
        has_stem_match = bool(query_tokens and any(stem in qt for stem in path_name_stems for qt in query_tokens))
        if symbol_info and (score >= DETAIL_THRESHOLD or any(t in path.lower() for t in query_tokens) or has_stem_match):
            line = f"{('  ' * (len(parts) - 1))}{parts[-1]}  {format_symbols_hint(symbol_info.get('symbols', []), None)}"
        else:
            line = f"{('  ' * (len(parts) - 1))}{parts[-1]}"
            
        if current_len + len(line) + 1 < max_chars:
            lines.append(line); current_len += len(line) + 1
        else:
            if not lines or "... (hard" not in lines[-1]: lines.append("... (tree exceeds token limit)")
            break
    return "\n".join(lines)


# =============================================================================
# Utility functions
# =============================================================================

def get_valid_paths_from_index(file_index: List[Dict]) -> set:
    return {entry["path"] for entry in file_index}


def filter_valid_paths(selected_paths: List[str], file_index: List[Dict]) -> List[str]:
    valid_paths = get_valid_paths_from_index(file_index)
    path_lookup: Dict[str, str] = {}
    for path in valid_paths:
        path_lookup[path] = path
        if path.startswith("packages/"):
            short_path = path[len("packages/"):]
            if short_path not in path_lookup: path_lookup[short_path] = path
    
    result, seen = [], set()
    for selected in selected_paths:
        cleaned = selected.strip().strip("/")
        if cleaned in path_lookup:
            actual = path_lookup[cleaned]
            if actual not in seen: result.append(actual); seen.add(actual)
            continue
        if not cleaned.startswith("packages/"):
            with_prefix = f"packages/{cleaned}"
            if with_prefix in path_lookup:
                actual = path_lookup[with_prefix]
                if actual not in seen: result.append(actual); seen.add(actual)
    return result