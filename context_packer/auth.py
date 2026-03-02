"""
API Key management and rate limiting.

Supports two modes:
1. Database mode (DATABASE_URL set): Keys and credits in Postgres
2. In-memory mode (fallback): Keys from API_KEYS env var

Pricing model:
- 100 free credits on signup
- $9 = 1000 credits (never expire)
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .config import settings


class Tier(str, Enum):
    """API tier."""
    FREE = "free"
    PAID = "paid"  # Has purchased credits


# Credits-based pricing: $9 = 1000 calls
CREDITS_PER_PURCHASE = 1000
PRICE_USD = 9
SIGNUP_BONUS_CREDITS = 100


# =============================================================================
# Database-backed auth (used when DATABASE_URL is set)
# =============================================================================

async def validate_key_db(key: str) -> Optional[dict]:
    """
    Validate API key against database.
    Returns key info with user tier and credits, or None if invalid.
    """
    from .db import get_api_key_info
    return await get_api_key_info(key)


async def check_credits_db(user_id: int) -> tuple[bool, int]:
    """
    Check if user has credits available.
    Returns (has_credits, current_balance).
    """
    from .db import get_user_credits
    credits = await get_user_credits(user_id)
    return (credits > 0, credits)


async def spend_credit_db(user_id: int, key_id: int) -> tuple[bool, int]:
    """
    Spend 1 credit for an API call.
    Returns (success, remaining_credits).
    """
    from .db import spend_credit
    return await spend_credit(user_id, key_id)


# =============================================================================
# In-memory auth (fallback when no DATABASE_URL)
# =============================================================================

@dataclass
class APIKey:
    """Represents an API key with credits and metadata."""
    key: str
    tier: Tier
    user_id: str
    credits: int = SIGNUP_BONUS_CREDITS
    created_at: float = field(default_factory=time.time)


class InMemoryKeyManager:
    """In-memory API key validation and credit tracking."""
    
    def __init__(self):
        self._keys: dict[str, APIKey] = {}
    
    def add_key(self, key: str, tier: Tier, user_id: str, credits: int = SIGNUP_BONUS_CREDITS) -> APIKey:
        """Add a new API key."""
        api_key = APIKey(key=key, tier=tier, user_id=user_id, credits=credits)
        self._keys[key] = api_key
        return api_key
    
    def validate_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key exists."""
        return self._keys.get(key)
    
    def check_credits(self, key: str) -> tuple[bool, int]:
        """Check if a key has credits available. Returns (has_credits, balance)."""
        api_key = self._keys.get(key)
        if not api_key:
            return (False, 0)
        return (api_key.credits > 0, api_key.credits)
    
    def spend_credit(self, key: str) -> tuple[bool, int]:
        """Spend 1 credit. Returns (success, remaining)."""
        api_key = self._keys.get(key)
        if not api_key or api_key.credits <= 0:
            return (False, api_key.credits if api_key else 0)
        api_key.credits -= 1
        return (True, api_key.credits)
    
    def add_credits(self, key: str, amount: int) -> int:
        """Add credits to a key. Returns new balance."""
        api_key = self._keys.get(key)
        if api_key:
            api_key.credits += amount
            return api_key.credits
        return 0


# Global in-memory manager
_memory_manager: Optional[InMemoryKeyManager] = None


def get_memory_manager() -> InMemoryKeyManager:
    """Get or create the in-memory key manager."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = InMemoryKeyManager()
    return _memory_manager


def init_keys_from_env(env_keys: str) -> None:
    """
    Initialize keys from environment variable.
    Format: "key1:tier1:user1,key2:tier2:user2" or "key1:tier1:user1:credits,..."
    """
    manager = get_memory_manager()
    
    if not env_keys:
        return
    
    for entry in env_keys.split(","):
        entry = entry.strip()
        if not entry:
            continue
        
        parts = entry.split(":")
        if len(parts) < 3:
            continue
        
        key, tier_str, user_id = parts[0], parts[1], parts[2]
        credits = int(parts[3]) if len(parts) > 3 else SIGNUP_BONUS_CREDITS
        
        try:
            tier = Tier(tier_str.lower())
        except ValueError:
            tier = Tier.FREE
        
        manager.add_key(key=key, tier=tier, user_id=user_id, credits=credits)


# =============================================================================
# Unified Auth Interface
# =============================================================================

def is_db_mode() -> bool:
    """Check if we're using database mode."""
    return bool(settings.DATABASE_URL)


async def validate_api_key(key: str) -> Optional[dict]:
    """
    Validate an API key. Works in both DB and in-memory mode.
    
    Returns dict with:
        - key_id (int, DB mode) or key (str, memory mode)
        - tier (str)
        - user_id (int or str)
        - credits (int)
    Or None if invalid.
    """
    if is_db_mode():
        info = await validate_key_db(key)
        if info:
            return {
                "key_id": info["key_id"],
                "tier": info["tier"],
                "user_id": info["user_id"],
                "credits": info["credits"],
            }
        return None
    else:
        manager = get_memory_manager()
        api_key = manager.validate_key(key)
        if api_key:
            return {
                "key": key,
                "tier": api_key.tier.value,
                "user_id": api_key.user_id,
                "credits": api_key.credits,
            }
        return None


async def check_credits(key: str, key_info: dict) -> tuple[bool, int]:
    """
    Check if user has credits available.
    Returns (has_credits, current_balance).
    """
    if is_db_mode():
        return await check_credits_db(key_info["user_id"])
    else:
        manager = get_memory_manager()
        return manager.check_credits(key)


async def spend_credit(key: str, key_info: dict) -> tuple[bool, int]:
    """
    Spend 1 credit for an API call.
    Returns (success, remaining_credits).
    """
    if is_db_mode():
        return await spend_credit_db(key_info["user_id"], key_info["key_id"])
    else:
        manager = get_memory_manager()
        return manager.spend_credit(key)
