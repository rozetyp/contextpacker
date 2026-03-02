"""
Git operations for benchmarks.
"""

import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional

# Cache directory
CACHE_DIR = Path(tempfile.gettempdir()) / "cp_eval_cache"
CACHE_DIR.mkdir(exist_ok=True)


def clone_repo(repo_url: str, github_pat: str = None) -> Optional[Path]:
    """Clone repo to cache. Returns path."""
    repo_hash = hashlib.md5(repo_url.encode()).hexdigest()[:12]
    repo_name = repo_url.rstrip("/").split("/")[-1]
    repo_path = CACHE_DIR / f"{repo_name}_{repo_hash}"
    
    if repo_path.exists():
        return repo_path
    
    # Build clone URL with PAT if provided
    if github_pat and "github.com" in repo_url:
        url_with_auth = repo_url.replace("https://", f"https://{github_pat}@")
    else:
        url_with_auth = repo_url
    
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url_with_auth, str(repo_path)],
            capture_output=True, timeout=120
        )
        return repo_path if repo_path.exists() else None
    except:
        return None


def get_code_files(repo_path: Path, max_files: int = 200) -> List[Tuple[str, str]]:
    """Get all code files with content. Returns [(path, content), ...]"""
    extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".rb"}
    files = []
    
    for f in repo_path.rglob("*"):
        if not f.is_file():
            continue
        if f.suffix not in extensions:
            continue
        if any(p in str(f) for p in ["node_modules", "__pycache__", ".git", "test", "tests", "examples"]):
            continue
        
        try:
            rel_path = str(f.relative_to(repo_path))
            content = f.read_text(errors="ignore")[:2000]
            files.append((rel_path, content))
        except:
            pass
    
    return files[:max_files]
