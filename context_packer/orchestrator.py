"""
Pack orchestrator - main pipeline for creating context packs.
Coordinates cloning, indexing, selection, packing, and cleanup.
"""

import asyncio
import shutil
import time
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

import tiktoken

from .config import settings
from .models import PackRequest, PackResponse, SkeletonResponse, FileItem, Stats
from .git_utils import (
    clone_repo_async,
    clone_private_repo,
    build_file_index,
    file_index_to_enriched_tree,
    filter_valid_paths,
    validate_repo_size,
    AuthenticationError,
    RepoNotFoundError,
)
from .selector import call_selector_llm, fallback_select
from .symbols import build_symbols_index
from .logging import logger
from .logging import log_pack_request
from .cache import file_index_cache
from .repo_cache import repo_cache


# =============================================================================
# Clone concurrency control
# =============================================================================

_clone_semaphore: Optional[asyncio.Semaphore] = None


def get_clone_semaphore() -> asyncio.Semaphore:
    """Get or create the global clone semaphore."""
    global _clone_semaphore
    if _clone_semaphore is None:
        _clone_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_CLONES)
    return _clone_semaphore


# =============================================================================
# Token counting
# =============================================================================

# Initialize tiktoken encoder (cl100k_base works for GPT-4, Claude, etc.)
try:
    ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    ENCODING = None


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    if ENCODING is None:
        # Fallback: rough approximation (4 chars per token)
        return len(text) // 4
    return len(ENCODING.encode(text))


# =============================================================================
# Language detection
# =============================================================================

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "jsx",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".java": "java",
    ".kt": "kotlin",
    ".scala": "scala",
    ".cs": "csharp",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".php": "php",
    ".swift": "swift",
    ".m": "objectivec",
    ".r": "r",
    ".R": "r",
    ".sql": "sql",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".ps1": "powershell",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sass": "sass",
    ".less": "less",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".dockerfile": "dockerfile",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".lua": "lua",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".clj": "clojure",
    ".dart": "dart",
    ".jl": "julia",
    ".nim": "nim",
    ".zig": "zig",
    ".v": "v",
    ".sol": "solidity",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
}


def guess_language(path: str) -> Optional[str]:
    """Guess programming language from file path."""
    # Check by extension
    suffix = Path(path).suffix.lower()
    if suffix in EXTENSION_TO_LANGUAGE:
        return EXTENSION_TO_LANGUAGE[suffix]
    
    # Check by filename
    name = Path(path).name.lower()
    if name == "dockerfile":
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    if name == "cmakelists.txt":
        return "cmake"
    if name in (".gitignore", ".dockerignore", ".npmignore"):
        return "gitignore"
    if name.endswith("rc") and not suffix:  # .bashrc, .zshrc, etc.
        return "bash"
    
    return None


# =============================================================================
# File reading
# =============================================================================

def read_file_safe(file_path: Path, max_size: int = 1024 * 1024) -> Optional[str]:
    """
    Safely read a file's content.
    
    Returns None if file can't be read, is too large, or is a symlink (security).
    """
    try:
        if not file_path.exists():
            return None
        
        # SECURITY: Prevent symlink traversal
        if file_path.is_symlink():
            return None
        
        size = file_path.stat().st_size
        if size > max_size:
            return None
        
        # Try UTF-8 first, then fall back to latin-1
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1")
    except Exception:
        return None


# =============================================================================
# Token estimation
# =============================================================================

def estimate_repo_tokens(
    file_index: List[dict],
    root_dir: Path,
    max_files_to_sample: int = 200
) -> int:
    """
    Estimate total tokens in the repository.
    
    Samples up to max_files_to_sample files and extrapolates.
    """
    if not file_index:
        return 0
    
    total_size = sum(entry["size_bytes"] for entry in file_index)
    sampled_tokens = 0
    sampled_bytes = 0
    
    # Sample files to get tokens-per-byte ratio
    for i, entry in enumerate(file_index):
        if i >= max_files_to_sample:
            break
        
        file_path = root_dir / entry["path"]
        content = read_file_safe(file_path)
        
        if content:
            sampled_tokens += count_tokens(content)
            sampled_bytes += entry["size_bytes"]
    
    if sampled_bytes == 0:
        # Fallback: assume ~3.5 bytes per token for code
        return int(total_size / 3.5)
    
    # Calculate ratio and extrapolate
    tokens_per_byte = sampled_tokens / sampled_bytes
    estimated_total = int(total_size * tokens_per_byte)
    
    return estimated_total


# =============================================================================
# Markdown building
# =============================================================================

def build_markdown(query: str, file_items: List[FileItem], contents: dict) -> str:
    """
    Build Markdown document with packed files.
    
    Includes self-describing metadata comments explaining why each file
    was selected, enabling downstream agents to understand context.
    
    Args:
        query: User's original question
        file_items: List of FileItem objects
        contents: Dict mapping path -> file content
    
    Returns:
        Formatted Markdown string with reason metadata
    """
    parts = [
        f"## Context for: {query}\n",
        f"*{len(file_items)} files selected*\n",
    ]
    
    for item in file_items:
        content = contents.get(item.path, "")
        lang = item.language or ""
        
        # Add self-describing metadata comment if reason available
        if item.reason:
            parts.append(f"\n<!-- contextpacker: reason=\"{item.reason}\" -->\n")
        parts.append(f"### {item.path}\n")
        parts.append(f"```{lang}\n{content}\n```\n")
    
    return "".join(parts)


# =============================================================================
# Main orchestrator
# =============================================================================

async def create_pack(request: PackRequest) -> PackResponse:
    """
    Create a context pack from a GitHub repository.
    
    This is the main pipeline that:
    1. Clones the repository (public or private with PAT)
    2. Builds file index
    3. Calls selector LLM
    4. Reads and packs selected files
    5. Computes statistics
    6. Cleans up (always for private repos, on error for public)
    
    Args:
        request: PackRequest with repo_url, query, optional vcs config
    
    Returns:
        PackResponse with markdown, files, and stats
    """
    # Generate unique pack ID and workspace directory
    pack_id = f"pack_{uuid.uuid4().hex[:12]}"
    work_dir: Optional[Path] = None
    repo_cached = False  # Track if we're using cached repo (don't delete)
    is_private_repo = request.vcs is not None and request.vcs.token
    
    total_start = time.perf_counter()
    response: Optional[PackResponse] = None
    error_msg: Optional[str] = None
    
    try:
        repo_url_str = str(request.repo_url)
        clone_ms = 0
        
        # =====================================================================
        # PRIVATE REPOS: Always use temp dir, never cache
        # =====================================================================
        if is_private_repo:
            # Create temp directory for private repo (will be cleaned up in finally)
            work_dir = Path(tempfile.mkdtemp(prefix="cp-private-"))
            
            clone_start = time.perf_counter()
            
            # Log without token (NEVER log token!)
            print(f"[INFO] Cloning private repo: {repo_url_str} (via PAT)")
            
            try:
                async with asyncio.timeout(settings.CLONE_QUEUE_TIMEOUT_S):
                    async with get_clone_semaphore():
                        await clone_private_repo(
                            repo_url=repo_url_str,
                            token=request.vcs.token,
                            target_dir=work_dir,
                            branch=request.vcs.branch,
                            timeout_s=settings.MAX_CLONE_TIMEOUT_S,
                        )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    f"QUEUE_TIMEOUT: Server busy, waited {settings.CLONE_QUEUE_TIMEOUT_S}s for clone slot."
                )
            
            clone_ms = int((time.perf_counter() - clone_start) * 1000)
            
            # Build file index (no caching for private repos)
            file_index = build_file_index(work_dir)
            
            if not file_index:
                raise RuntimeError("EMPTY_REPO: No indexable files found")
            
            validate_repo_size(file_index)
            files_considered = len(file_index)
            
            # Build symbol index for enriched tree
            symbols_index = build_symbols_index(work_dir, file_index, max_files=500)
            tree_text = file_index_to_enriched_tree(file_index, symbols_index, query=request.query)
            
            tokens_raw_repo = estimate_repo_tokens(file_index, work_dir)
            cache_hit = False
        
        # =====================================================================
        # PUBLIC REPOS: Use caching as before
        # =====================================================================
        else:
            work_dir = Path(f"/tmp/cp_{pack_id}")
            
            # Check repo cache first (persistent disk cache)
            if settings.REPO_CACHE_ENABLED:
                cached_path = repo_cache.get(repo_url_str)
                if cached_path is not None:
                    work_dir = cached_path
                    repo_cached = True
                    clone_ms = 0
            
            # Check in-memory file index cache
            cached = file_index_cache.get(repo_url_str)
            cache_hit = cached is not None and repo_cached
            
            if cached:
                file_index = cached.file_index
                # Rebuild tree per-query: same repo, different queries weight files differently
                tree_text = file_index_to_enriched_tree(file_index, cached.symbols_index, query=request.query)
                tokens_raw_repo = cached.tokens_raw_estimate
                files_considered = len(file_index)
            else:
                # Clone repository if not in repo cache
                if not repo_cached:
                    clone_start = time.perf_counter()
                    temp_clone_dir = Path(f"/tmp/cp_{pack_id}")
                    
                    try:
                        queue_start = time.perf_counter()
                        async with asyncio.timeout(settings.CLONE_QUEUE_TIMEOUT_S):
                            async with get_clone_semaphore():
                                await clone_repo_async(
                                    repo_url=repo_url_str,
                                    target_dir=temp_clone_dir,
                                    github_token=settings.GITHUB_PAT,
                                    timeout_s=settings.MAX_CLONE_TIMEOUT_S,
                                )
                    except asyncio.TimeoutError:
                        raise RuntimeError(
                            f"QUEUE_TIMEOUT: Server busy, waited {settings.CLONE_QUEUE_TIMEOUT_S}s for clone slot."
                        )
                    
                    clone_ms = int((time.perf_counter() - clone_start) * 1000)
                    
                    if settings.REPO_CACHE_ENABLED:
                        work_dir = repo_cache.put(repo_url_str, temp_clone_dir)
                        repo_cached = True
                    else:
                        work_dir = temp_clone_dir
                
                # Build file index and validate size
                file_index = build_file_index(work_dir)
                
                if not file_index:
                    raise RuntimeError("EMPTY_REPO: No indexable files found")
                
                validate_repo_size(file_index)
                files_considered = len(file_index)
                
                # Build symbol index for enriched tree
                symbols_index = build_symbols_index(work_dir, file_index, max_files=500)
                tree_text = file_index_to_enriched_tree(file_index, symbols_index, query=request.query)
                
                tokens_raw_repo = estimate_repo_tokens(file_index, work_dir)
                
                # Cache for next time (public repos only)
                file_index_cache.set(
                    repo_url_str,
                    file_index,
                    symbols_index,
                    tokens_raw_repo,
                )
        
        # =====================================================================
        # 7. Call selector LLM
        # =====================================================================
        selector_start = time.perf_counter()
        
        # Extract file paths for lexical matching
        all_file_paths = [entry["path"] for entry in file_index]
        
        selected_paths, selector_cost, _, reasons_dict = await call_selector_llm(
            file_tree_text=tree_text,
            query=request.query,
            max_files=10,
            include_reasons=True,
            file_paths=all_file_paths,
        )
        
        selector_ms = int((time.perf_counter() - selector_start) * 1000)
        
        # Filter to only valid paths
        selected_paths = filter_valid_paths(selected_paths, file_index)
        
        # =====================================================================
        # 6. Fallback if selector returned nothing useful
        # =====================================================================
        if not selected_paths:
            selected_paths = fallback_select(file_index, request.query, max_files=5)
        
        # =====================================================================
        # 7. Read selected files and build pack (ADAPTIVE PACKING STRATEGY)
        # =====================================================================
        packing_start = time.perf_counter()
        
        # Token budget tracking
        TOKEN_BUFFER = 50  # Reserve for markdown overhead (headers, fences)
        max_tokens = request.max_tokens
        
        # Start with header
        header = f"## Context for: {request.query}\n"
        header_tokens = count_tokens(header)
        available_budget = max_tokens - header_tokens - TOKEN_BUFFER
        
        file_items: List[FileItem] = []
        contents: dict = {}
        files_truncated_paths: List[str] = []
        files_content_truncated: List[str] = []
        
        # =====================================================================
        # ADAPTIVE PACKING: Read all files and calculate sizes first
        # =====================================================================
        file_data = []  # (path, content, language, raw_tokens)
        loop = asyncio.get_running_loop()
        
        for path in selected_paths:
            file_path = work_dir / path
            # Read file in thread pool to avoid blocking event loop
            content = await loop.run_in_executor(None, read_file_safe, file_path)
            
            if content is None:
                continue
            
            language = guess_language(path) or ""
            file_block = f"\n### {path}\n```{language}\n{content}\n```\n"
            file_tokens = count_tokens(file_block)
            
            file_data.append((path, content, language, file_tokens))
        
        # Calculate total size if all files were included full
        total_full_tokens = sum(tokens for _, _, _, tokens in file_data)
        
        # If everything fits, just pack it all
        if total_full_tokens <= available_budget:
            for path, content, language, tokens in file_data:
                file_items.append(FileItem(
                    path=path,
                    tokens=tokens,
                    language=language if language else None,
                    relevance_score=None,
                    reason=reasons_dict.get(path),
                ))
                contents[path] = content
        else:
            # LLM-GUIDED ADAPTIVE PACKING
            # Use priority levels from selector LLM to allocate budget intelligently
            # Priority weights: critical (55%), important (30%), supplementary (10%), reference (5%)
            
            priority_weights = {
                "critical": 0.55,
                "important": 0.30,
                "supplementary": 0.10,
                "reference": 0.05
            }
            
            # Group files by LLM-assigned priority
            files_by_priority = {"critical": [], "important": [], "supplementary": [], "reference": []}
            
            for path, content, lang, tokens in file_data:
                # Extract priority from reasons_dict (if LLM provided it)
                reason_data = reasons_dict.get(path)
                if isinstance(reason_data, dict) and "priority" in reason_data:
                    priority = reason_data["priority"]
                else:
                    # Default: first file is important, rest are supplementary
                    priority = "important" if path == file_data[0][0] else "supplementary"
                
                if priority in files_by_priority:
                    files_by_priority[priority].append((path, content, lang, tokens))
                else:
                    files_by_priority["important"].append((path, content, lang, tokens))
            
            # Allocate budget based on priority levels
            for priority in ["critical", "important", "supplementary", "reference"]:
                files_in_priority = files_by_priority[priority]
                if not files_in_priority:
                    continue
                
                priority_budget = int(available_budget * priority_weights[priority])
                total_size_in_priority = sum(tokens for _, _, _, tokens in files_in_priority)
                
                # If all files in this priority fit fully within budget, pack them full
                if total_size_in_priority <= priority_budget:
                    for path, content, lang, tokens in files_in_priority:
                        file_items.append(FileItem(
                            path=path,
                            tokens=tokens,
                            language=lang if lang else None,
                            relevance_score=None,
                            reason=reasons_dict.get(path),
                        ))
                        contents[path] = content
                else:
                    # Proportionally truncate files within this priority
                    for path, content, lang, full_tokens in files_in_priority:
                        allocated = int((full_tokens / total_size_in_priority) * priority_budget)
                        allocated = max(100, min(allocated, full_tokens))  # Min 100 tokens
                        
                        if full_tokens <= allocated:
                            # File fits fully
                            file_items.append(FileItem(
                                path=path,
                                tokens=full_tokens,
                                language=lang if lang else None,
                                relevance_score=None,
                                reason=reasons_dict.get(path),
                            ))
                            contents[path] = content
                        else:
                            # Fast optimized truncation: Tokenize once, slice, decode
                            # Estimate content budget by subtracting header/footer overhead
                            overhead_block = f"\n### {path}\n```{lang}\n\n```\n"
                            overhead_tokens = count_tokens(overhead_block)
                            content_budget = max(50, allocated - overhead_tokens)
                            
                            if ENCODING:
                                # Precise truncation
                                all_tokens = ENCODING.encode(content)
                                if len(all_tokens) <= content_budget:
                                    truncated_content = content
                                else:
                                    # Keep tokens
                                    kept_tokens = all_tokens[:content_budget]
                                    truncated_content = ENCODING.decode(kept_tokens)
                                    # Try to cut at last newline for readability
                                    last_newline = truncated_content.rfind('\n')
                                    if last_newline > len(truncated_content) * 0.8: # Only if it doesn't lose too much
                                        truncated_content = truncated_content[:last_newline]
                            else:
                                # Fallback char-based truncation
                                char_budget = content_budget * 4
                                truncated_content = content[:char_budget]
                                last_newline = truncated_content.rfind('\n')
                                if last_newline > 0:
                                    truncated_content = truncated_content[:last_newline]
                            
                            truncated_content += f"\n# ... (file truncated, {full_tokens - allocated} tokens omitted)"
                            
                            final_block = f"\n### {path}\n```{lang}\n{truncated_content}\n```\n"
                            final_tokens = count_tokens(final_block)
                            
                            file_items.append(FileItem(
                                path=path,
                                tokens=final_tokens,
                                language=lang if lang else None,
                                relevance_score=None,
                                reason=reasons_dict.get(path),
                            ))
                            contents[path] = truncated_content
                            files_content_truncated.append(path)
        
        # Build markdown from included files only
        markdown = build_markdown(request.query, file_items, contents)
        
        # Add truncation notice if needed
        notices = []
        if files_content_truncated:
            notices.append(f"*{len(files_content_truncated)} files partially included (truncated to fit budget)*")
        if files_truncated_paths:
            notices.append(f"*{len(files_truncated_paths)} files omitted to stay under {max_tokens} token limit*")
        if notices:
            markdown += "\n---\n" + "\n".join(notices) + "\n"
        
        tokens_packed = count_tokens(markdown)
        
        packing_ms = int((time.perf_counter() - packing_start) * 1000)
        
        # =====================================================================
        # 8. Compute statistics
        # =====================================================================
        tokens_saved = max(tokens_raw_repo - tokens_packed, 0)
        
        # Keep selector_cost for internal logging, but don't expose in response
        internal_selector_cost = round(selector_cost, 6)
        
        stats = Stats(
            tokens_packed=tokens_packed,
            tokens_raw_repo=tokens_raw_repo,
            tokens_saved=tokens_saved,
            files_selected=len(file_items),
            files_considered=files_considered,
            repo_clone_ms=clone_ms,
            selector_ms=selector_ms,
            packing_ms=packing_ms,
            truncated=len(files_truncated_paths) > 0,
            files_truncated=len(files_truncated_paths),
            cache_hit=cache_hit,
        )
        # Set private attribute for internal logging (excluded from JSON response)
        stats._internal_selector_cost = internal_selector_cost
        
        # =====================================================================
        # 9. Build response
        # =====================================================================
        response = PackResponse(
            id=pack_id,
            engine_version=settings.ENGINE_VERSION,
            markdown=markdown,
            files=file_items,
            stats=stats,
        )
        
        return response
    
    except Exception as e:
        error_msg = str(e)
        raise
    
    finally:
        # =====================================================================
        # Log request (success or failure)
        # =====================================================================
        duration_ms = int((time.perf_counter() - total_start) * 1000)
        # Include internal cost in log dict (excluded from API response)
        stats_for_log = {}
        if response:
            stats_for_log = response.stats.model_dump()
            stats_for_log["_internal_selector_cost"] = response.stats._internal_selector_cost
        log_pack_request(
            pack_id=pack_id,
            repo_url=str(request.repo_url),
            query=request.query,
            stats=stats_for_log,
            error=error_msg,
            duration_ms=duration_ms,
        )
        
        # =====================================================================
        # Cleanup workspace
        # SECURITY: Always delete private repos, even on error
        # =====================================================================
        if work_dir and work_dir.exists():
            # Private repos: ALWAYS delete (never cache user's private code)
            if is_private_repo:
                shutil.rmtree(work_dir, ignore_errors=True)
            # Public repos: Only delete if not cached
            elif not repo_cached:
                shutil.rmtree(work_dir, ignore_errors=True)


async def create_skeleton(request: PackRequest, force_refresh: bool = False) -> SkeletonResponse:
    """
    Generate a Semantic Skeleton for a repository.
    Skips the Selector LLM and Packing steps.
    """
    pack_id = f"skel_{uuid.uuid4().hex[:12]}"
    work_dir: Optional[Path] = None
    repo_cached = False
    is_private_repo = request.vcs is not None and request.vcs.token
    
    total_start = time.perf_counter()
    response: Optional[SkeletonResponse] = None
    error_msg: Optional[str] = None
    
    try:
        repo_url_str = str(request.repo_url)
        clone_ms = 0
        
        # --- Shared Logic: Clone / Cache ---
        # (This duplication is tech debt, but we keep it safe for now to avoid refactoring risk)
        
        if is_private_repo:
            work_dir = Path(tempfile.mkdtemp(prefix="cp-private-"))
            clone_start = time.perf_counter()
            print(f"[INFO] Skeleton: Cloning private repo: {repo_url_str}")
            try:
                async with asyncio.timeout(settings.CLONE_QUEUE_TIMEOUT_S):
                    async with get_clone_semaphore():
                        await clone_private_repo(
                            repo_url=repo_url_str,
                            token=request.vcs.token,
                            target_dir=work_dir,
                            branch=request.vcs.branch,
                            timeout_s=settings.MAX_CLONE_TIMEOUT_S,
                        )
            except asyncio.TimeoutError:
                raise RuntimeError("QUEUE_TIMEOUT")
            clone_ms = int((time.perf_counter() - clone_start) * 1000)
            
            file_index = build_file_index(work_dir)
            if not file_index: raise RuntimeError("EMPTY_REPO")
            validate_repo_size(file_index)
            
            symbols_index = build_symbols_index(work_dir, file_index, max_files=500)
            tree_text = file_index_to_enriched_tree(file_index, symbols_index, query=request.query)
            tokens_raw_repo = estimate_repo_tokens(file_index, work_dir)
            files_considered = len(file_index)
            cache_hit = False

        else: # Public
            work_dir = Path(f"/tmp/cp_{pack_id}")
            if settings.REPO_CACHE_ENABLED:
                cached_path = repo_cache.get(repo_url_str)
                if cached_path is not None:
                    work_dir = cached_path
                    repo_cached = True
                    clone_ms = 0
            
            cached = file_index_cache.get(repo_url_str)
            cache_hit = cached is not None and repo_cached and not force_refresh
            
            if cache_hit:
                # Rebuild tree per-query using cached symbols for accurate relevance weighting
                tree_text = file_index_to_enriched_tree(cached.file_index, cached.symbols_index, query=request.query)
                tokens_raw_repo = cached.tokens_raw_estimate
                files_considered = len(cached.file_index)
            else:
                if not repo_cached or force_refresh:
                    clone_start = time.perf_counter()
                    temp_clone_dir = Path(f"/tmp/cp_clone_{pack_id}")
                    try:
                        async with asyncio.timeout(settings.CLONE_QUEUE_TIMEOUT_S):
                            async with get_clone_semaphore():
                                await clone_repo_async(
                                    repo_url=repo_url_str,
                                    target_dir=temp_clone_dir,
                                    github_token=settings.GITHUB_PAT,
                                    timeout_s=settings.MAX_CLONE_TIMEOUT_S,
                                )
                    except asyncio.TimeoutError:
                        raise RuntimeError("QUEUE_TIMEOUT")
                    clone_ms = int((time.perf_counter() - clone_start) * 1000)
                    
                    if settings.REPO_CACHE_ENABLED:
                        work_dir = repo_cache.put(repo_url_str, temp_clone_dir)
                        repo_cached = True
                    else:
                        work_dir = temp_clone_dir
                
                file_index = build_file_index(work_dir)
                if not file_index: raise RuntimeError("EMPTY_REPO")
                validate_repo_size(file_index)
                
                symbols_index = build_symbols_index(work_dir, file_index, max_files=500)
                tree_text = file_index_to_enriched_tree(file_index, symbols_index, query=request.query)
                tokens_raw_repo = estimate_repo_tokens(file_index, work_dir)
                files_considered = len(file_index)
                
                file_index_cache.set(repo_url_str, file_index, symbols_index, tokens_raw_repo)

        # --- Return Skeleton ---
        stats = Stats(
            tokens_packed=count_tokens(tree_text), # Tree size
            tokens_raw_repo=tokens_raw_repo,
            tokens_saved=0,
            files_selected=0,
            files_considered=files_considered,
            repo_clone_ms=clone_ms,
            selector_ms=0,
            packing_ms=0,
            cache_hit=cache_hit,
        )
        
        response = SkeletonResponse(
            id=pack_id,
            repo_url=repo_url_str,
            tree=tree_text,
            stats=stats
        )
        return response

    except Exception as e:
        error_msg = str(e)
        raise
    
    finally:
        # Cleanup (Same as create_pack)
        if work_dir and work_dir.exists():
            if is_private_repo:
                shutil.rmtree(work_dir, ignore_errors=True)
            elif not repo_cached:
                shutil.rmtree(work_dir, ignore_errors=True)
