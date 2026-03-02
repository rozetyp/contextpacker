"""
Database connection and schema management.
Uses asyncpg for async Postgres access.
"""

import asyncpg
import secrets
from typing import Optional
from datetime import datetime, timezone

from .config import settings

# Connection pool (initialized on startup)
_pool: Optional[asyncpg.Pool] = None


# =============================================================================
# Schema
# =============================================================================

SCHEMA_SQL = """
-- Users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    stripe_customer_id VARCHAR(255),
    tier VARCHAR(20) NOT NULL DEFAULT 'free',
    credits INTEGER NOT NULL DEFAULT 100,  -- Prepaid credits (100 on signup)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id SERIAL PRIMARY KEY,
    key VARCHAR(60) NOT NULL UNIQUE,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);

-- Usage tracking (per key per month) - kept for analytics
CREATE TABLE IF NOT EXISTS usage (
    id SERIAL PRIMARY KEY,
    api_key_id INTEGER NOT NULL REFERENCES api_keys(id) ON DELETE CASCADE,
    month VARCHAR(7) NOT NULL,  -- '2025-11' format
    count INTEGER NOT NULL DEFAULT 0,
    UNIQUE(api_key_id, month)
);

-- Credit transactions (audit trail)
CREATE TABLE IF NOT EXISTS credit_transactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    amount INTEGER NOT NULL,  -- positive=add, negative=spend
    type VARCHAR(20) NOT NULL,  -- 'signup_bonus', 'purchase', 'usage', 'refund'
    stripe_payment_id VARCHAR(255),  -- for purchases
    note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_key ON api_keys(key);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_usage_key_month ON usage(api_key_id, month);
CREATE INDEX IF NOT EXISTS idx_credit_tx_user ON credit_transactions(user_id);
"""


# =============================================================================
# Connection Management
# =============================================================================

async def init_db() -> None:
    """Initialize database connection pool and create schema."""
    global _pool
    
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL not set")
    
    _pool = await asyncpg.create_pool(
        settings.DATABASE_URL,
        min_size=2,
        max_size=10,
    )
    
    # Create schema
    async with _pool.acquire() as conn:
        await conn.execute(SCHEMA_SQL)


async def close_db() -> None:
    """Close database connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Get the connection pool."""
    if _pool is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _pool


# =============================================================================
# Helpers
# =============================================================================

def generate_api_key() -> str:
    """Generate a new API key."""
    return f"cp_live_{secrets.token_hex(20)}"


def get_current_month() -> str:
    """Get current month in YYYY-MM format."""
    return datetime.now(timezone.utc).strftime("%Y-%m")


# =============================================================================
# User Operations
# =============================================================================

async def create_user(email: str, tier: str = "free", initial_credits: int = 100) -> int:
    """Create a new user with signup bonus credits. Returns user ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Create user with initial credits
            row = await conn.fetchrow(
                "INSERT INTO users (email, tier, credits) VALUES ($1, $2, $3) RETURNING id",
                email, tier, initial_credits
            )
            user_id = row["id"]
            
            # Record signup bonus transaction
            if initial_credits > 0:
                await conn.execute("""
                    INSERT INTO credit_transactions (user_id, amount, type, note)
                    VALUES ($1, $2, 'signup_bonus', 'Welcome bonus')
                """, user_id, initial_credits)
            
            return user_id


async def get_user_by_email(email: str) -> Optional[dict]:
    """Get user by email."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, stripe_customer_id, tier, credits, created_at FROM users WHERE email = $1",
            email
        )
        return dict(row) if row else None


async def update_user_tier(user_id: int, tier: str) -> None:
    """Update a user's tier."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET tier = $1 WHERE id = $2",
            tier, user_id
        )


async def update_user_stripe_customer(user_id: int, stripe_customer_id: str) -> None:
    """Update a user's Stripe customer ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET stripe_customer_id = $1 WHERE id = $2",
            stripe_customer_id, user_id
        )


# =============================================================================
# API Key Operations
# =============================================================================

async def create_api_key(user_id: int, name: str = None) -> str:
    """Create a new API key for a user. Returns the key."""
    pool = get_pool()
    key = generate_api_key()
    
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO api_keys (key, user_id, name) VALUES ($1, $2, $3)",
            key, user_id, name
        )
    
    return key


async def get_api_key_info(key: str) -> Optional[dict]:
    """
    Get API key info including user tier and credits.
    Returns None if key is invalid or revoked.
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT 
                ak.id as key_id,
                ak.user_id,
                ak.name as key_name,
                ak.created_at as key_created_at,
                u.email,
                u.tier,
                u.credits
            FROM api_keys ak
            JOIN users u ON ak.user_id = u.id
            WHERE ak.key = $1 AND ak.revoked_at IS NULL
        """, key)
        
        return dict(row) if row else None


async def revoke_api_key(key: str) -> bool:
    """Revoke an API key. Returns True if key was found and revoked."""
    pool = get_pool()
    
    async with pool.acquire() as conn:
        result = await conn.execute(
            "UPDATE api_keys SET revoked_at = NOW() WHERE key = $1 AND revoked_at IS NULL",
            key
        )
        return result == "UPDATE 1"


async def list_user_keys(user_id: int) -> list[dict]:
    """List all API keys for a user."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, key, name, created_at, revoked_at
            FROM api_keys
            WHERE user_id = $1
            ORDER BY created_at DESC
        """, user_id)
        return [dict(row) for row in rows]


# =============================================================================
# Usage Operations
# =============================================================================

async def get_usage(api_key_id: int, month: str = None) -> int:
    """Get usage count for an API key for a given month."""
    pool = get_pool()
    month = month or get_current_month()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT count FROM usage WHERE api_key_id = $1 AND month = $2",
            api_key_id, month
        )
        return row["count"] if row else 0


async def increment_usage(api_key_id: int) -> int:
    """Increment usage count for current month. Returns new count."""
    pool = get_pool()
    month = get_current_month()
    
    async with pool.acquire() as conn:
        # Upsert: insert or increment
        row = await conn.fetchrow("""
            INSERT INTO usage (api_key_id, month, count)
            VALUES ($1, $2, 1)
            ON CONFLICT (api_key_id, month)
            DO UPDATE SET count = usage.count + 1
            RETURNING count
        """, api_key_id, month)
        return row["count"]


async def get_user_total_usage(user_id: int, month: str = None) -> int:
    """Get total usage across all keys for a user for a given month."""
    pool = get_pool()
    month = month or get_current_month()
    
    async with pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT COALESCE(SUM(u.count), 0) as total
            FROM usage u
            JOIN api_keys ak ON u.api_key_id = ak.id
            WHERE ak.user_id = $1 AND u.month = $2
        """, user_id, month)
        return row["total"]


# =============================================================================
# Credit Operations
# =============================================================================

async def get_user_credits(user_id: int) -> int:
    """Get current credit balance for a user."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT credits FROM users WHERE id = $1",
            user_id
        )
        return row["credits"] if row else 0


async def spend_credit(user_id: int, api_key_id: int) -> tuple[bool, int]:
    """
    Spend 1 credit for an API call.
    Returns (success, remaining_credits).
    Also records usage in the usage table for analytics.
    """
    pool = get_pool()
    month = get_current_month()
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Atomically decrement credit if available
            row = await conn.fetchrow("""
                UPDATE users 
                SET credits = credits - 1 
                WHERE id = $1 AND credits > 0
                RETURNING credits
            """, user_id)
            
            if row is None:
                # No credits available
                credits = await conn.fetchval(
                    "SELECT credits FROM users WHERE id = $1", user_id
                )
                return (False, credits or 0)
            
            remaining = row["credits"]
            
            # Record usage for analytics (month-based)
            await conn.execute("""
                INSERT INTO usage (api_key_id, month, count)
                VALUES ($1, $2, 1)
                ON CONFLICT (api_key_id, month)
                DO UPDATE SET count = usage.count + 1
            """, api_key_id, month)
            
            return (True, remaining)


async def add_credits(
    user_id: int, 
    amount: int, 
    tx_type: str = "purchase",
    stripe_payment_id: str = None,
    note: str = None
) -> int:
    """
    Add credits to a user account.
    Returns new credit balance.
    """
    pool = get_pool()
    
    async with pool.acquire() as conn:
        async with conn.transaction():
            # Add credits
            row = await conn.fetchrow("""
                UPDATE users 
                SET credits = credits + $1 
                WHERE id = $2
                RETURNING credits
            """, amount, user_id)
            
            if row is None:
                raise ValueError(f"User {user_id} not found")
            
            new_balance = row["credits"]
            
            # Record transaction
            await conn.execute("""
                INSERT INTO credit_transactions 
                    (user_id, amount, type, stripe_payment_id, note)
                VALUES ($1, $2, $3, $4, $5)
            """, user_id, amount, tx_type, stripe_payment_id, note)
            
            return new_balance


async def get_user_by_stripe_customer(stripe_customer_id: str) -> Optional[dict]:
    """Get user by Stripe customer ID."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, email, stripe_customer_id, tier, credits, created_at FROM users WHERE stripe_customer_id = $1",
            stripe_customer_id
        )
        return dict(row) if row else None


async def get_credit_transactions(user_id: int, limit: int = 20) -> list[dict]:
    """Get recent credit transactions for a user."""
    pool = get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, amount, type, stripe_payment_id, note, created_at
            FROM credit_transactions
            WHERE user_id = $1
            ORDER BY created_at DESC
            LIMIT $2
        """, user_id, limit)
        return [dict(row) for row in rows]
