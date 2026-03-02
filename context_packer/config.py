"""
Configuration management for ContextPacker.
Uses pydantic-settings to load from environment variables and .env file.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================================================
    # Required settings
    # ==========================================================================
    
    # LLM API key (Gemini, OpenAI, or Anthropic)
    LLM_API_KEY: str
    
    # Database URL (Postgres)
    DATABASE_URL: Optional[str] = None
    
    # API authentication token for incoming requests (legacy single-key mode)
    API_AUTH_TOKEN: Optional[str] = None
    
    # API keys with tiers: "key1:tier:user,key2:tier:user"
    # Example: "cp_live_abc:pro:user_1,cp_live_xyz:free:user_2"
    # Only used if DATABASE_URL is not set (in-memory fallback)
    API_KEYS: Optional[str] = None
    
    # ==========================================================================
    # Optional settings with defaults
    # ==========================================================================
    
    # GitHub PAT for avoiding rate limits (highly recommended)
    GITHUB_PAT: Optional[str] = None
    
    # LLM provider: "gemini", "openai", or "anthropic"
    LLM_PROVIDER: str = "gemini"
    
    # Model for file selection
    LLM_MODEL: str = "gemini-2.5-flash"
    
    # Git clone timeout in seconds
    MAX_CLONE_TIMEOUT_S: int = 15
    
    # Maximum concurrent git clones
    MAX_CONCURRENT_CLONES: int = 5
    
    # Clone queue timeout (how long to wait for semaphore)
    CLONE_QUEUE_TIMEOUT_S: int = 30
    
    # Maximum characters for file tree (50k allows ~1500 files to be visible)
    MAX_TREE_CHARS: int = 50000
    
    # Maximum directory depth to index (6 to handle packages/*/src/*/file.js patterns)
    MAX_TREE_DEPTH: int = 6
    
    # Maximum single file size in bytes (2MB default)
    MAX_FILE_SIZE_BYTES: int = 2 * 1024 * 1024
    
    # Repo content cache settings
    REPO_CACHE_MAX_SIZE_GB: float = 10.0  # Max disk space for cached repos
    REPO_CACHE_ENABLED: bool = True  # Enable/disable repo caching
    
    # Stripe settings (for credit purchases)
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    STRIPE_PRICE_ID: Optional[str] = None  # Price ID for $9/1000 credits
    
    # Server port
    PORT: int = 8000
    
    # Engine version (LLM-guided adaptive packing)
    ENGINE_VERSION: str = "0.2.0"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()
