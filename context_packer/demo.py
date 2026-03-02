"""
Demo endpoint for ContextPacker landing page.

Compares 3 retrieval approaches: ContextPacker, Embeddings, BM25.
No authentication required, limited to whitelisted repos.
"""

import asyncio
import hashlib
import math
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field

from .git_utils import clone_repo_async
from .models import PackRequest
from .orchestrator import create_pack


# =============================================================================
# Demo config
# =============================================================================

# 25 popular repos for demo - normalized URLs for lookup
DEMO_REPOS = {
    # Python
    "https://github.com/pallets/flask",
    "https://github.com/tiangolo/fastapi",
    "https://github.com/encode/starlette",
    "https://github.com/neoteroi/blacksheep",
    "https://github.com/django/django",
    "https://github.com/pydantic/pydantic",
    "https://github.com/psf/requests",
    "https://github.com/pallets/click",
    "https://github.com/strawberry-graphql/strawberry",
    "https://github.com/encode/httpx",
    # JavaScript/Node
    "https://github.com/expressjs/express",
    "https://github.com/koajs/koa",
    "https://github.com/fastify/fastify",
    "https://github.com/honojs/hono",
    "https://github.com/trpc/trpc",
    "https://github.com/nestjs/nest",
    "https://github.com/typeorm/typeorm",
    "https://github.com/prisma/prisma",
    # Go
    "https://github.com/gin-gonic/gin",
    "https://github.com/gofiber/fiber",
    "https://github.com/labstack/echo",
    # Rust
    "https://github.com/tokio-rs/axum",
    "https://github.com/actix/actix-web",
    # AI
    "https://github.com/langchain-ai/langchain",
    "https://github.com/openai/openai-python",
}

# Rate limiting - stricter for open demo
DEMO_RATE_LIMIT_WHITELISTED = 10  # requests per minute per IP for whitelisted repos
DEMO_RATE_LIMIT_CUSTOM = 3        # requests per minute per IP for custom repos
DEMO_HOURLY_LIMIT_CUSTOM = 10     # max custom repo requests per hour per IP
_rate_limiter: dict[str, list[float]] = defaultdict(list)
_hourly_limiter: dict[str, list[float]] = defaultdict(list)

# Limits for custom repos (stricter than whitelisted)
DEMO_MAX_REPO_SIZE_MB = 100       # Max repo size for custom demo
DEMO_MAX_FILES = 10_000           # Max files for custom demo


# =============================================================================
# BM25 Implementation (industry-standard lexical baseline)
# =============================================================================

def bm25_tokenize(text: str) -> List[str]:
    """Tokenize text for BM25: split camelCase, snake_case, extract words."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25:
    """BM25 ranking for code search - the industry-standard lexical baseline."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: List[Tuple[str, List[str]]] = []
        self.doc_freqs: Counter = Counter()
        self.avg_dl = 0.0
        self.N = 0
    
    def index(self, documents: List[Tuple[str, str]]):
        """Index documents: [(path, content), ...]"""
        self.docs = []
        self.doc_freqs = Counter()
        
        for path, content in documents:
            tokens = bm25_tokenize(path) + bm25_tokenize(content)
            self.docs.append((path, tokens))
            for term in set(tokens):
                self.doc_freqs[term] += 1
        
        self.N = len(self.docs)
        self.avg_dl = sum(len(d[1]) for d in self.docs) / self.N if self.N else 1
    
    def score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """BM25 score for a single document."""
        score = 0.0
        doc_len = len(doc_tokens)
        term_freqs = Counter(doc_tokens)
        
        for term in query_tokens:
            if term not in term_freqs:
                continue
            tf = term_freqs[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl))
            score += idf * tf_component
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for query, return [(path, score), ...]"""
        query_tokens = bm25_tokenize(query)
        scored = [(path, self.score(query_tokens, doc_tokens)) 
                  for path, doc_tokens in self.docs]
        scored = [(p, s) for p, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


# =============================================================================
# Models
# =============================================================================

class DemoRequest(BaseModel):
    """Request body for POST /v1/demo"""
    repo_url: str = Field(..., description="Any public GitHub repository URL (e.g., https://github.com/owner/repo)")
    question: str = Field(..., min_length=1, max_length=500, description="Question about the codebase")
    max_tokens: int = Field(default=6000, ge=1000, le=16000, description="Max context tokens (1000-16000)")


class DemoResponse(BaseModel):
    """Response from demo endpoint with comparison"""
    question: str
    repo: str
    repo_url: str
    contextpacker: dict  # {answer, files_used, tokens, latency_ms}
    embeddings: dict     # {answer, files_used, tokens, latency_ms}
    bm25: dict           # {answer, files_used, tokens, latency_ms}


# =============================================================================
# Router
# =============================================================================

router = APIRouter(tags=["Demo"])


def _check_rate_limit(client_ip: str, is_custom_repo: bool) -> bool:
    """
    IP-based rate limiting. Stricter limits for custom (non-whitelisted) repos.
    Returns True if allowed.
    """
    now = time.time()
    
    # Per-minute limit
    window = 60
    limit = DEMO_RATE_LIMIT_CUSTOM if is_custom_repo else DEMO_RATE_LIMIT_WHITELISTED
    _rate_limiter[client_ip] = [t for t in _rate_limiter[client_ip] if now - t < window]
    if len(_rate_limiter[client_ip]) >= limit:
        return False
    
    # Additional hourly limit for custom repos
    if is_custom_repo:
        hourly_window = 3600
        _hourly_limiter[client_ip] = [t for t in _hourly_limiter[client_ip] if now - t < hourly_window]
        if len(_hourly_limiter[client_ip]) >= DEMO_HOURLY_LIMIT_CUSTOM:
            return False
        _hourly_limiter[client_ip].append(now)
    
    _rate_limiter[client_ip].append(now)
    return True


def _is_valid_github_url(url: str) -> bool:
    """Check if URL is a valid public GitHub repo URL."""
    pattern = r'^https://github\.com/[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+/?$'
    return bool(re.match(pattern, url))


@router.post(
    "/v1/demo",
    response_model=DemoResponse,
    summary="Live demo comparison",
    description="Compare ContextPacker vs Embeddings vs BM25. Works with any public GitHub repo. No API key required."
)
async def demo_endpoint(request: DemoRequest, req: Request):
    """
    Live demo endpoint for the landing page.
    
    - Works with ANY public GitHub repo (with stricter rate limits for custom repos)
    - Whitelisted repos have higher rate limits
    - No authentication required
    - Returns 3-way comparison: ContextPacker, Embeddings, BM25
    """
    # Normalize and validate repo URL
    repo_url = request.repo_url.lower().rstrip('/')
    
    # Check if valid GitHub URL
    if not _is_valid_github_url(repo_url):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INVALID_URL", "message": "Please provide a valid public GitHub repo URL (e.g., https://github.com/owner/repo)"}
        )
    
    is_whitelisted = repo_url in DEMO_REPOS
    
    # Rate limit by IP (stricter for custom repos)
    client_ip = req.client.host if req.client else "unknown"
    if not _check_rate_limit(client_ip, is_custom_repo=not is_whitelisted):
        if is_whitelisted:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "RATE_LIMITED", "message": "Too many demo requests. Try again in a minute."}
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={"error": "RATE_LIMITED", "message": "Too many custom repo requests. Limit: 3/min, 10/hour. Try a whitelisted repo or wait."}
            )
    
    question = request.question
    max_tokens = request.max_tokens
    
    # Get API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not openai_key:
        raise HTTPException(status_code=500, detail={"error": "CONFIG_ERROR", "message": "Demo not configured"})
    
    repo_name = repo_url.split('/')[-1]
    
    # Helper to call OpenAI
    async def get_llm_answer(context: str, system_prompt: str) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {openai_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}" if context else f"Question about the {repo_name} repository: {question}"}
                    ],
                    "max_tokens": 800,
                    "temperature": 0.3,
                }
            )
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "Error")
    
    # Clone repo once
    url_hash = hashlib.md5(repo_url.encode()).hexdigest()[:12]
    temp_base = Path(tempfile.gettempdir()) / "demo_repos"
    temp_base.mkdir(parents=True, exist_ok=True)
    work_dir = temp_base / f"{repo_name}_{url_hash}"
    
    try:
        if not work_dir.exists() or not (work_dir / ".git").exists():
            await clone_repo_async(repo_url, work_dir)
    except Exception as e:
        error_msg = str(e).lower()
        if "not found" in error_msg or "404" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "REPO_NOT_FOUND", "message": "Repository not found. Make sure it's a public GitHub repo."}
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "CLONE_FAILED", "message": f"Failed to clone repo: {str(e)[:100]}"}
        )
    
    # For custom repos, validate size limits
    if not is_whitelisted:
        total_size = 0
        file_count = 0
        for file_path in work_dir.rglob("*"):
            if file_path.is_file() and ".git" not in str(file_path):
                total_size += file_path.stat().st_size
                file_count += 1
        
        total_mb = total_size / (1024 * 1024)
        if total_mb > DEMO_MAX_REPO_SIZE_MB:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "REPO_TOO_LARGE", "message": f"Repo is {total_mb:.0f}MB (demo limit: {DEMO_MAX_REPO_SIZE_MB}MB). Get an API key for larger repos."}
            )
        if file_count > DEMO_MAX_FILES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "REPO_TOO_LARGE", "message": f"Repo has {file_count:,} files (demo limit: {DEMO_MAX_FILES:,}). Get an API key for larger repos."}
            )
    
    # Find source files (shared by embeddings and BM25)
    skip_dirs = {".git", ".github", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", 
                 "test", "tests", "docs", "examples", "benchmarks", "scripts", "release", "tools"}
    source_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java"}
    source_files: List[Tuple[str, str]] = []
    for file_path in work_dir.rglob("*"):
        if file_path.is_dir() or file_path.suffix.lower() not in source_exts:
            continue
        parts = file_path.relative_to(work_dir).parts
        if any(part in skip_dirs for part in parts):
            continue
        try:
            content = file_path.read_text(errors="ignore")
            source_files.append((str(file_path.relative_to(work_dir)), content))
        except Exception:
            continue
    
    # Define async functions for each approach
    async def run_contextpacker() -> dict:
        start = time.time()
        try:
            pack_request = PackRequest(repo_url=repo_url, query=question, max_tokens=max_tokens)
            pack_response = await create_pack(pack_request)
            answer = await get_llm_answer(
                pack_response.markdown,
                "You are a helpful coding assistant. Answer based on the provided code context. Be concise but thorough."
            )
            return {
                "answer": answer,
                "files_used": [f.path for f in pack_response.files],
                "tokens": pack_response.stats.tokens_packed,
                "latency_ms": int((time.time() - start) * 1000)
            }
        except Exception as e:
            return {"answer": f"ContextPacker error: {str(e)}", "files_used": [], "tokens": 0, "latency_ms": int((time.time() - start) * 1000)}
    
    async def run_embeddings() -> dict:
        if not gemini_key:
            return {"answer": "Embeddings not configured", "files_used": [], "tokens": 0, "latency_ms": 0}
        start = time.time()
        try:
            import numpy as np
            import tiktoken
            
            files_to_embed = [(path, content[:1500]) for path, content in source_files]
            
            async def get_embeddings_batch(texts: list) -> list:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key={gemini_key}"
                requests = [{"model": "models/gemini-embedding-001", "content": {"parts": [{"text": t}]}, "taskType": "RETRIEVAL_DOCUMENT", "outputDimensionality": 768} for t in texts]
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(url, json={"requests": requests})
                    return [np.array(e["values"]) for e in resp.json().get("embeddings", [])]
            
            async def get_query_embedding(query_text: str) -> "np.ndarray":
                url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent?key={gemini_key}"
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(url, json={"content": {"parts": [{"text": query_text}]}, "taskType": "RETRIEVAL_QUERY", "outputDimensionality": 768})
                    return np.array(resp.json()["embedding"]["values"])
            
            texts = [f"File: {p}\n{c}" for p, c in files_to_embed[:200]]
            embeddings = []
            for i in range(0, len(texts), 100):
                embeddings.extend(await get_embeddings_batch(texts[i:i+100]))
            
            query_emb = await get_query_embedding(question)
            similarities = [(path, float(np.dot(query_emb / np.linalg.norm(query_emb), emb / np.linalg.norm(emb)))) 
                           for (path, _), emb in zip(files_to_embed[:len(embeddings)], embeddings)]
            similarities.sort(key=lambda x: x[1], reverse=True)
            selected_paths = [p for p, _ in similarities[:50]]
            
            enc = tiktoken.encoding_for_model("gpt-4o")
            context_parts, tokens_used = [], 0
            for path in selected_paths[:20]:
                if tokens_used >= max_tokens:
                    break
                try:
                    content = (work_dir / path).read_text(errors="replace")
                    ext = Path(path).suffix.lower()
                    lang = {".py": "python", ".js": "javascript", ".ts": "typescript"}.get(ext, "")
                    block = f"### {path}\n```{lang}\n{content}\n```\n\n"
                    block_tokens = len(enc.encode(block))
                    if tokens_used + block_tokens <= max_tokens:
                        context_parts.append(block)
                        tokens_used += block_tokens
                except Exception:
                    continue
            
            answer = await get_llm_answer(''.join(context_parts), "You are a helpful coding assistant. Answer based on the provided code context. Be concise but thorough.")
            return {"answer": answer, "files_used": selected_paths[:10], "tokens": tokens_used, "latency_ms": int((time.time() - start) * 1000)}
        except Exception as e:
            return {"answer": f"Embeddings error: {str(e)}", "files_used": [], "tokens": 0, "latency_ms": int((time.time() - start) * 1000)}
    
    async def run_bm25() -> dict:
        start = time.time()
        try:
            import tiktoken
            bm25 = BM25()
            bm25.index(source_files)
            bm25_ranked = bm25.search(question, top_k=50)
            bm25_paths = [path for path, _ in bm25_ranked]
            
            enc = tiktoken.encoding_for_model("gpt-4o")
            context_parts, tokens_used = [], 0
            for path in bm25_paths[:20]:
                if tokens_used >= max_tokens:
                    break
                try:
                    content = (work_dir / path).read_text(errors="replace")
                    ext = Path(path).suffix.lower()
                    lang = {".py": "python", ".js": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust"}.get(ext, "")
                    block = f"### {path}\n```{lang}\n{content}\n```\n\n"
                    block_tokens = len(enc.encode(block))
                    if tokens_used + block_tokens <= max_tokens:
                        context_parts.append(block)
                        tokens_used += block_tokens
                except Exception:
                    continue
            
            answer = await get_llm_answer(''.join(context_parts), "You are a helpful coding assistant. Answer based on the provided code context. Be concise but thorough.")
            return {"answer": answer, "files_used": bm25_paths[:10], "tokens": tokens_used, "latency_ms": int((time.time() - start) * 1000)}
        except Exception as e:
            return {"answer": f"BM25 error: {str(e)}", "files_used": [], "tokens": 0, "latency_ms": int((time.time() - start) * 1000)}
    
    # Run all 3 approaches in parallel
    cp_result, emb_result, bm25_result = await asyncio.gather(
        run_contextpacker(),
        run_embeddings(),
        run_bm25()
    )
    
    return DemoResponse(
        question=question,
        repo=repo_name,
        repo_url=repo_url,
        contextpacker=cp_result,
        embeddings=emb_result,
        bm25=bm25_result,
    )
