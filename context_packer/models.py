"""
Pydantic models for API request/response schemas.
"""

from pydantic import BaseModel, Field, HttpUrl, PrivateAttr
from typing import List, Literal, Optional, Union


# =============================================================================
# VCS Configuration (for private repos)
# =============================================================================

class VCSConfig(BaseModel):
    """
    Version control system configuration for private repository access.
    
    Currently supports GitHub via Personal Access Token (PAT).
    Future: github_enterprise, gitlab, bitbucket.
    
    SECURITY:
    - Token is never logged, never persisted
    - Used only for clone, discarded after request
    """
    
    provider: Literal["github"] = Field(
        default="github",
        description="VCS provider. Currently only 'github' supported."
    )
    
    token: str = Field(
        ...,
        min_length=1,
        description="Personal Access Token for authentication. Never logged or stored."
    )
    
    branch: Optional[str] = Field(
        default=None,
        description="Branch to clone. Defaults to repo's default branch."
    )


class PackRequest(BaseModel):
    """Request body for POST /v1/packs"""
    
    repo_url: HttpUrl = Field(
        ...,
        description="GitHub HTTPS URL (e.g., https://github.com/owner/repo)",
        examples=["https://github.com/expressjs/express"]
    )
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Natural language question about the codebase",
        examples=["How does the routing system work?"]
    )
    
    max_tokens: int = Field(
        default=12000,
        ge=100,
        le=100000,
        description="Target maximum tokens for packed context"
    )
    
    vcs: Optional[VCSConfig] = Field(
        default=None,
        description="VCS config for private repos. Include token for authentication."
    )


class FileItem(BaseModel):
    """Individual file in the pack response"""
    
    path: str = Field(
        ...,
        description="Relative path from repo root"
    )
    
    tokens: int = Field(
        ...,
        ge=0,
        description="Token count for this file"
    )
    
    language: Optional[str] = Field(
        default=None,
        description="Detected programming language"
    )
    
    relevance_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Relevance score from selector (0-1)"
    )
    
    reason: Optional[Union[str, dict]] = Field(
        default=None,
        description="Brief explanation of why this file was selected (string or dict with reason and priority)"
    )


class Stats(BaseModel):
    """Statistics about the pack operation"""
    
    tokens_packed: int = Field(
        ...,
        description="Total tokens in the packed context"
    )
    
    tokens_raw_repo: int = Field(
        ...,
        description="Estimated total tokens in the raw repository"
    )
    
    tokens_saved: int = Field(
        ...,
        description="Tokens saved (raw - packed)"
    )
    
    files_selected: int = Field(
        ...,
        description="Number of files included in pack"
    )
    
    files_considered: int = Field(
        ...,
        description="Total files in repository index"
    )
    
    repo_clone_ms: int = Field(
        ...,
        description="Time to clone repository in milliseconds"
    )
    
    selector_ms: int = Field(
        ...,
        description="Time for selector LLM call in milliseconds"
    )
    
    packing_ms: int = Field(
        ...,
        description="Time to read and pack files in milliseconds"
    )
    
    truncated: bool = Field(
        default=False,
        description="Whether pack was truncated to fit max_tokens"
    )
    
    files_truncated: int = Field(
        default=0,
        description="Number of files omitted due to token limit"
    )
    
    cache_hit: bool = Field(
        default=False,
        description="Whether file index was served from cache"
    )
    
    # Private attribute - excluded from JSON response, used for internal logging
    _internal_selector_cost: float = PrivateAttr(default=0.0)


class PackResponse(BaseModel):
    """Response with packed markdown context."""
    id: str
    engine_version: str
    markdown: str
    files: List[FileItem]
    stats: Stats


class SkeletonResponse(BaseModel):
    """Response with repository file tree skeleton."""
    id: str
    repo_url: str
    tree: str
    stats: Stats


class ErrorResponse(BaseModel):
    """Standard error response format"""
    
    error: str = Field(
        ...,
        description="Error code (e.g., REPO_CLONE_TIMEOUT)",
        examples=["REPO_CLONE_TIMEOUT", "INVALID_URL", "UNAUTHORIZED"]
    )
    
    message: str = Field(
        ...,
        description="Human-readable error message"
    )


class SignupRequest(BaseModel):
    """Request body for POST /signup"""
    
    email: str = Field(
        ...,
        description="Email address for API key",
        examples=["user@example.com"]
    )


class SignupResponse(BaseModel):
    """Response for POST /signup"""
    
    api_key: str = Field(
        ...,
        description="Generated API key",
        examples=["cp_live_abc123..."]
    )
    
    credits: int = Field(
        ...,
        description="Number of free credits",
        examples=[100]
    )


class HealthResponse(BaseModel):
    """Response for GET /health"""
    
    status: str = Field(
        default="ok",
        description="Service status"
    )
    
    engine_version: str = Field(
        ...,
        description="Engine version"
    )
