"""
FastAPI application entrypoint for ContextPacker.
"""

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Request, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .config import settings
from .models import (
    PackRequest,
    PackResponse,
    SkeletonResponse,
    SignupRequest,
    SignupResponse,
    HealthResponse,
    ErrorResponse,
)
from .orchestrator import create_pack, create_skeleton
from .git_utils import (
    AuthenticationError,
    RepoNotFoundError,
    CloneTimeoutError,
    CloneFailedError,
)
from .auth import (
    is_db_mode,
    validate_api_key,
    check_credits,
    spend_credit,
    init_keys_from_env,
    get_memory_manager,
    Tier,
)
from .db import (
    get_user_by_email,
    create_user,
    create_api_key,
)


# =============================================================================
# Timeout Middleware
# =============================================================================

REQUEST_TIMEOUT_S = 60


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce request timeout."""
    
    async def dispatch(self, request: Request, call_next):
        try:
            return await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_S)
        except asyncio.TimeoutError:
            return JSONResponse(
                status_code=504,
                content={"error": "REQUEST_TIMEOUT", "message": f"Request timed out after {REQUEST_TIMEOUT_S}s"}
            )


# =============================================================================
# App initialization
# =============================================================================

app = FastAPI(
    title="ContextPacker API",
    description="Transform any GitHub repo into optimized AI context in seconds. "
                "We intelligently select and pack relevant files into token-efficient Markdown.",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(TimeoutMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Include routers
# =============================================================================

from .demo import router as demo_router
from .billing import router as billing_router

app.include_router(demo_router)
app.include_router(billing_router)


# =============================================================================
# Initialize on startup
# =============================================================================

@app.on_event("startup")
async def startup():
    """Initialize database tables and in-memory keys on startup."""
    if is_db_mode():
        from .db import init_db
        await init_db()
        print("[AUTH] Running in DATABASE mode (Postgres)")
    else:
        init_keys_from_env(settings.API_KEYS or "")
        mgr = get_memory_manager()
        print(f"[AUTH] Running in MEMORY mode ({len(mgr._keys)} keys loaded)")


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the service is running and show current stats."
)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        engine_version="0.2.0",
    )


@app.post(
    "/signup",
    response_model=SignupResponse,
    tags=["Auth"],
    summary="Get API key",
    description="Create account and get API key instantly. If email exists, regenerates key."
)
async def signup_endpoint(request: SignupRequest):
    """Create user and generate API key."""
    if not is_db_mode():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"error": "SERVICE_UNAVAILABLE", "message": "Signup not available in memory mode"}
        )
    
    try:
        # Check if user exists
        existing_user = await get_user_by_email(request.email)
        
        if existing_user:
            # User exists - create new API key
            user_id = existing_user["id"]
            api_key = await create_api_key(user_id, "Regenerated key")
            credits = existing_user["credits"]
        else:
            # New user - create user and API key
            user_id = await create_user(request.email, "free", 100)
            api_key = await create_api_key(user_id, "Signup key")
            credits = 100
        
        return SignupResponse(
            api_key=api_key,
            credits=credits
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": "SIGNUP_FAILED", "message": f"Failed to create account: {str(e)}"}
        )


# =============================================================================
# Exception handlers
# =============================================================================

@app.exception_handler(AuthenticationError)
async def auth_error_handler(request: Request, exc: AuthenticationError):
    """Handle authentication failures for private repos."""
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={
            "error": "AUTHENTICATION_FAILED",
            "message": str(exc),
            "hint": "For private repos, include a valid GitHub PAT in vcs.token"
        }
    )


@app.exception_handler(RepoNotFoundError)
async def repo_not_found_handler(request: Request, exc: RepoNotFoundError):
    """Handle repo not found errors."""
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={
            "error": "REPO_NOT_FOUND",
            "message": str(exc),
            "hint": "Check that the repository exists and is accessible"
        }
    )


@app.exception_handler(asyncio.TimeoutError)
async def timeout_handler(request: Request, exc: asyncio.TimeoutError):
    """Handle timeout errors from subprocess operations."""
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={
            "error": "TIMEOUT",
            "message": "Operation timed out. The repository may be too large.",
            "hint": "Try a smaller repository or contact support for enterprise repos"
        }
    )


@app.exception_handler(CloneTimeoutError)
async def clone_timeout_handler(request: Request, exc: CloneTimeoutError):
    """Handle clone timeout errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "REPO_CLONE_TIMEOUT",
            "message": "Repository clone took too long. It might be too large.",
            "hint": "Try a smaller repository or ensure it isn't an asset dump."
        }
    )


@app.exception_handler(CloneFailedError)
async def clone_failed_handler(request: Request, exc: CloneFailedError):
    """Handle general clone failure errors."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "REPO_CLONE_FAILED",
            "message": str(exc),
            "hint": "Ensure the URL is a valid public GitHub repository."
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler."""
    error_msg = str(exc)
    
    # Don't expose internal errors in production
    if "token" in error_msg.lower() or "key" in error_msg.lower():
        error_msg = "An internal error occurred"
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "INTERNAL_ERROR",
            "message": error_msg
        }
    )


# =============================================================================
# Debug endpoint
# =============================================================================

@app.get("/debug/tree/{repo_hash}")
async def debug_tree(repo_hash: str):
    """Debug: show cached tree_text for a repo."""
    from .cache import file_index_cache
    for key, entry in file_index_cache._cache.items():
        if repo_hash in key or key in repo_hash:
            return {
                "key": key,
                "tree_text_preview": entry.tree_text[:3000],
                "tree_text_len": len(entry.tree_text),
                "file_count": len(entry.file_index),
                "has_app_py": any(f["path"] == "src/flask/app.py" for f in entry.file_index),
                "flask_files": [f["path"] for f in entry.file_index if "flask" in f["path"]][:20],
            }
    return {"error": "not found", "keys": list(file_index_cache._cache.keys())}


# =============================================================================
# Main API endpoint
# =============================================================================

@app.post(
    "/v1/packs",
    response_model=PackResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Unauthorized"},
        422: {"model": ErrorResponse, "description": "Processing error"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    tags=["Packs"],
    summary="Create a context pack",
    description="Clone a GitHub repository, intelligently select relevant files, and pack them into Markdown context."
)
async def create_pack_endpoint(
    request: PackRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
):
    """
    Create a context pack from a GitHub repository.
    
    This endpoint:
    1. Validates the repository URL (GitHub only for v0.1)
    2. Clones the repository (shallow, with timeout)
    3. Uses an LLM to select relevant files based on your query
    4. Packs selected files into Markdown format
    5. Returns the pack with token savings statistics
    """
    # Authentication & Credit Check
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "UNAUTHORIZED", "message": "Missing API key. Include X-API-Key header."}
        )
    
    key_info = await validate_api_key(x_api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "UNAUTHORIZED", "message": "Invalid API key."}
        )
    
    has_credits, balance = await check_credits(x_api_key, key_info)
    if not has_credits:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "NO_CREDITS", "message": "No credits remaining. Top up at contextpacker.com/pricing", "credits": balance},
            headers={"X-Credits-Remaining": "0"}
        )
    
    # URL validation
    repo_url_str = str(request.repo_url)
    if not repo_url_str.startswith("https://github.com/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INVALID_URL", "message": "Only public GitHub repositories are supported. URL must start with https://github.com/"}
        )
    
    path_parts = repo_url_str.replace("https://github.com/", "").rstrip("/").split("/")
    if len(path_parts) < 2 or not path_parts[0] or not path_parts[1]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "INVALID_URL", "message": "Invalid GitHub URL format. Expected: https://github.com/owner/repo"}
        )
    
    # Create pack
    response = await create_pack(request)
    
    # Spend credit after success
    await spend_credit(x_api_key, key_info)
    
    return response


@app.post(
    "/v1/skeleton",
    response_model=SkeletonResponse,
    tags=["Packs"],
    summary="Get repository file tree",
    description="Get the optimized, query-aware file tree (Semantic Skeleton) without packing file contents. Ideal for agents planning their own navigation."
)
async def create_skeleton_endpoint(
    request: PackRequest,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    x_force_refresh: str | None = Header(default=None, alias="X-Force-Refresh"),
):
    """Generate repository skeleton."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key")
    
    key_info = await validate_api_key(x_api_key)
    if not key_info:
        raise HTTPException(status_code=401, detail="Invalid API key")
        
    has_credits, _ = await check_credits(x_api_key, key_info)
    if not has_credits:
        raise HTTPException(status_code=429, detail="No credits remaining")

    # Reuse PackRequest for simplicity (repo_url, query)
    force_refresh = str(x_force_refresh).lower() == "true"
    response = await create_skeleton(request, force_refresh=force_refresh)
    
    # Skeleton calls are cheaper (0.2 credits?) - for now charge 1 credit
    await spend_credit(x_api_key, key_info)
    
    return response


# =============================================================================
# Static files & Landing page
# =============================================================================

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/", include_in_schema=False)
async def serve_landing():
    """Serve the landing page at root."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({
        "message": "ContextPacker API",
        "docs": "/docs",
        "debug_static_path": str(index_path),
        "static_exists": STATIC_DIR.exists()
    })


@app.get("/how-it-works", include_in_schema=False)
@app.get("/how-it-works.html", include_in_schema=False)
async def serve_how_it_works():
    """Serve How It Works page."""
    path = STATIC_DIR / "how-it-works.html"
    if path.exists():
        return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/evaluation", include_in_schema=False)
@app.get("/evaluation.html", include_in_schema=False)
async def serve_evaluation():
    """Serve Evaluation page."""
    path = STATIC_DIR / "evaluation.html"
    if path.exists():
        return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/alternatives", include_in_schema=False)
@app.get("/alternatives.html", include_in_schema=False)
async def serve_alternatives():
    """Serve Alternatives page."""
    path = STATIC_DIR / "alternatives.html"
    if path.exists():
        return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Page not found")


@app.get("/docs.html", include_in_schema=False)
async def serve_docs_page():
    """Serve API Docs page."""
    path = STATIC_DIR / "docs.html"
    if path.exists():
        return FileResponse(path, media_type="text/html")
    raise HTTPException(status_code=404, detail="Page not found")


# Static assets (favicon, logo, manifest, etc.)
@app.get("/favicon.svg", include_in_schema=False)
async def serve_favicon():
    return FileResponse(STATIC_DIR / "favicon.svg", media_type="image/svg+xml")


@app.get("/logo.svg", include_in_schema=False)
async def serve_logo():
    return FileResponse(STATIC_DIR / "logo.svg", media_type="image/svg+xml")


@app.get("/apple-touch-icon.svg", include_in_schema=False)
async def serve_apple_touch_icon():
    return FileResponse(STATIC_DIR / "apple-touch-icon.svg", media_type="image/svg+xml")


@app.get("/og-image.svg", include_in_schema=False)
async def serve_og_image():
    return FileResponse(STATIC_DIR / "og-image.svg", media_type="image/svg+xml")


@app.get("/manifest.json", include_in_schema=False)
async def serve_manifest():
    return FileResponse(STATIC_DIR / "manifest.json", media_type="application/json")


@app.get("/robots.txt", include_in_schema=False)
async def serve_robots():
    return FileResponse(STATIC_DIR / "robots.txt", media_type="text/plain")


@app.get("/sitemap.xml", include_in_schema=False)
async def serve_sitemap():
    return FileResponse(STATIC_DIR / "sitemap.xml", media_type="application/xml")


# =============================================================================
# Run with: uvicorn context_packer.main:app --reload --port 8000
# =============================================================================
