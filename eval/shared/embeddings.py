"""
Embeddings baseline with file-level indexing.

Simple, fair comparison for our use case:
1. INDEXING: Clone repo → embed each file (path + content) → cache to disk
2. QUERY: Embed query → cosine similarity → return top K files

This matches what we're comparing against: file-level RAG without infrastructure.
We're NOT trying to beat function-level CodeSearchNet - we're showing we match
file-level embeddings without the vector DB setup.
"""

import hashlib
import json
import math
import time
import httpx
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict

from .clone import CACHE_DIR, clone_repo

# Embedding cache directory (persistent)
EMBEDDINGS_DIR = CACHE_DIR / "embeddings_v3"  # v3 = file-level, better coverage
EMBEDDINGS_DIR.mkdir(exist_ok=True)

# Gemini embedding model
GEMINI_KEY = "your_gemini_api_key_here"  # Free tier key
MODEL = "text-embedding-004"


@dataclass
class EmbeddingIndex:
    """Pre-computed embedding index for a repo (file-level)."""
    repo_url: str
    repo_hash: str
    version: str  # "v3" = file-level with full coverage
    files: List[str]  # file paths
    embeddings: List[List[float]]  # corresponding embeddings
    indexed_at: str
    stats: Dict  # {files_found, files_embedded}


def _get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from Gemini API."""
    try:
        resp = httpx.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:embedContent?key={GEMINI_KEY}",
            json={"model": f"models/{MODEL}", "content": {"parts": [{"text": text[:8000]}]}},
            timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["embedding"]["values"]
    except Exception as e:
        print(f"    Embedding error: {e}")
    return None


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _index_path(repo_url: str) -> Path:
    """Get path to cached index file."""
    repo_hash = hashlib.md5(repo_url.encode()).hexdigest()[:12]
    return EMBEDDINGS_DIR / f"{repo_hash}.json"


def get_all_code_files(repo_path: Path, max_files: int = 300) -> List[Tuple[str, str]]:
    """
    Get ALL code files with content - no filtering except noise dirs.
    
    Unlike the old approach, we:
    - Include test files (fair comparison)
    - Include examples
    - Use more content (3000 chars vs 200)
    """
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb"}
    noise_dirs = {"node_modules", "__pycache__", ".git", "vendor", "dist", ".venv", "venv"}
    
    files = []
    for f in repo_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix not in extensions:
            continue
        # Only skip actual noise, not tests/examples
        if any(p in f.parts for p in noise_dirs):
            continue
        
        try:
            rel_path = str(f.relative_to(repo_path))
            content = f.read_text(errors="ignore")[:3000]  # More content than before
            files.append((rel_path, content))
        except:
            pass
    
    # Sort by path for consistency
    files.sort(key=lambda x: x[0])
    return files[:max_files]


def index_repo(repo_url: str, github_pat: str = "", force: bool = False) -> Optional[EmbeddingIndex]:
    """
    Pre-index a repo with FILE-LEVEL embeddings. Cached to disk.
    
    Each file is one embedding: "File: path\n<content>"
    """
    index_file = _index_path(repo_url)
    
    # Return cached if exists
    if index_file.exists() and not force:
        try:
            data = json.loads(index_file.read_text())
            if data.get("version") == "v3":  # File-level v3
                print(f"    ✓ Loaded cached index ({len(data['files'])} files)")
                return EmbeddingIndex(**data)
        except:
            pass
    
    print(f"    Indexing {repo_url} (file-level)...")
    
    # Clone repo
    repo_path = clone_repo(repo_url, github_pat)
    if not repo_path:
        print(f"    ✗ Failed to clone")
        return None
    
    # Get files
    files = get_all_code_files(repo_path)
    if not files:
        print(f"    ✗ No code files found")
        return None
    
    print(f"    Embedding {len(files)} files...")
    
    # Embed each file
    file_paths = []
    embeddings = []
    for i, (path, content) in enumerate(files):
        # Format: path + content (like original approach but more content)
        text = f"File: {path}\n{content}"
        emb = _get_embedding(text)
        if emb:
            file_paths.append(path)
            embeddings.append(emb)
        
        if (i + 1) % 50 == 0:
            print(f"    ... {i + 1}/{len(files)}")
            time.sleep(1)  # Rate limiting for Gemini free tier
    
    if not embeddings:
        print(f"    ✗ No embeddings generated")
        return None
    
    # Create index
    from datetime import datetime
    index = EmbeddingIndex(
        repo_url=repo_url,
        repo_hash=hashlib.md5(repo_url.encode()).hexdigest()[:12],
        version="v3",
        files=file_paths,
        embeddings=embeddings,
        indexed_at=datetime.now().isoformat(),
        stats={
            "files_found": len(files),
            "files_embedded": len(embeddings)
        }
    )
    
    # Save to disk
    index_file.write_text(json.dumps(asdict(index)))
    print(f"    ✓ Indexed {len(embeddings)} files → {index_file.name}")
    
    return index


def search(repo_url: str, query: str, top_k: int = 10, github_pat: str = "") -> Tuple[List[str], float]:
    """
    Search using pre-indexed FILE-LEVEL embeddings.
    """
    start = time.perf_counter()
    
    # Load or create index
    index = index_repo(repo_url, github_pat)
    if not index:
        return [], (time.perf_counter() - start) * 1000
    
    # Embed query
    query_emb = _get_embedding(query)
    if not query_emb:
        return [], (time.perf_counter() - start) * 1000
    
    # Compute similarities
    scored = []
    for path, file_emb in zip(index.files, index.embeddings):
        score = _cosine_similarity(query_emb, file_emb)
        scored.append((path, score))
    
    # Sort and return top K
    scored.sort(key=lambda x: x[1], reverse=True)
    latency = (time.perf_counter() - start) * 1000
    
    return [p for p, _ in scored[:top_k]], latency


def preindex_all(repos: List[str], github_pat: str = ""):
    """Pre-index all benchmark repos. Run once before benchmarks."""
    print(f"Pre-indexing {len(repos)} repos (file-level)...")
    for repo_url in repos:
        print(f"\n{repo_url}")
        index_repo(repo_url, github_pat, force=True)
    print("\n✓ All repos indexed!")


# CLI for pre-indexing
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-index repos for file-level embedding search")
    parser.add_argument("--repo", help="Single repo URL to index")
    parser.add_argument("--all", action="store_true", help="Index all benchmark repos")
    parser.add_argument("--force", action="store_true", help="Force re-index")
    args = parser.parse_args()
    
    if args.repo:
        index_repo(args.repo, force=args.force)
    elif args.all:
        # Get all repos from benchmark files
        import os
        questions_dir = Path(__file__).parent.parent / "retrieval" / "questions"
        repos = set()
        for f in questions_dir.rglob("*.json"):
            try:
                data = json.loads(f.read_text())
                repos.add(data.get("repo_url", ""))
            except:
                pass
        repos.discard("")
        preindex_all(list(repos), os.getenv("GITHUB_PAT", ""))
