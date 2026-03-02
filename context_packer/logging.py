"""
Structured JSON logging for ContextPacker.
All request/response events are logged in a parseable format for SLO monitoring.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

# Configure JSON logger
logger = logging.getLogger("contextpacker")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_pack_request(
    pack_id: str,
    repo_url: str,
    query: str,
    stats: Dict[str, Any],
    error: Optional[str] = None,
    duration_ms: int = 0,
) -> None:
    """
    Emit structured log for a pack request.
    
    Args:
        pack_id: Unique pack identifier
        repo_url: Repository URL
        query: User's query
        stats: Stats dict from response (or partial stats if error)
        error: Error message if request failed
        duration_ms: Total request duration
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "pack_request",
        "pack_id": pack_id,
        "repo_url": repo_url,
        "query_length": len(query),
        "duration_ms": duration_ms,
        "error": error,
        # Flatten common stats for easy querying
        "tokens_packed": stats.get("tokens_packed", 0),
        "tokens_raw_repo": stats.get("tokens_raw_repo", 0),
        "tokens_saved": stats.get("tokens_saved", 0),
        "files_selected": stats.get("files_selected", 0),
        "files_truncated": stats.get("files_truncated", 0),
        "truncated": stats.get("truncated", False),
        "selector_llm_cost_usd": stats.get("_internal_selector_cost", 0),
        "clone_ms": stats.get("repo_clone_ms", 0),
        "selector_ms": stats.get("selector_ms", 0),
        "packing_ms": stats.get("packing_ms", 0),
    }
    logger.info(json.dumps(log_entry))


def log_metric(
    name: str,
    value: float,
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """
    Emit a metric event for monitoring.
    
    Args:
        name: Metric name (e.g., "request_latency_ms")
        value: Metric value
        tags: Optional tags for grouping
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "metric",
        "name": name,
        "value": value,
        "tags": tags or {},
    }
    logger.info(json.dumps(log_entry))


def log_error(
    error_code: str,
    message: str,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Emit an error event.
    
    Args:
        error_code: Error code (e.g., "REPO_CLONE_TIMEOUT")
        message: Human-readable message
        context: Additional context
    """
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": "error",
        "error_code": error_code,
        "message": message,
        "context": context or {},
    }
    logger.error(json.dumps(log_entry))
