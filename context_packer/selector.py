"""
LLM-based file selector for ContextPacker.
Uses a cheap LLM to intelligently select relevant files based on user query.
"""

import json
import re
import time
from typing import List, Tuple, Dict, Set

import httpx

from .config import settings


# =============================================================================
# Lexical path matching (minimal implementation)
# =============================================================================

def extract_query_terms(query: str) -> Set[str]:
    """Extract meaningful terms from query for path matching."""
    terms = set(re.findall(r'[a-z]+', query.lower()))
    # Keep terms with 3+ chars, exclude stopwords
    stopwords = {'the', 'how', 'does', 'what', 'where', 'which', 'when', 'why', 
                 'and', 'for', 'with', 'this', 'that', 'from', 'are', 'was', 'can',
                 'has', 'have', 'its', 'work', 'works', 'implement', 'implemented',
                 'blacksheep', 'flask', 'react', 'django', 'fastapi'}  # Common repo names
    return {t for t in terms if len(t) >= 3 and t not in stopwords}


# Extensions that are source code (should be prioritized in lexical hints)
SOURCE_CODE_EXTENSIONS = {
    '.py', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs',
    '.go', '.rs', '.java', '.kt', '.scala', '.rb', '.php',
    '.c', '.cpp', '.cc', '.h', '.hpp', '.cs', '.swift',
}

# Directories to exclude from lexical hints (output, not source)
LEXICAL_EXCLUDE_DIRS = {
    'results', 'output', 'outputs', 'dist', 'build', 'coverage',
    'docs', 'doc', 'documentation', 'docs_src', 'doc_src',
    'examples', 'example', 'samples', 'sample',
    'tests', 'test', '__tests__', 'spec', 'specs',
    '.github', 'scripts', 'script',
    # i18n data and DB migrations are data/infra, not source logic
    'locale', 'locales', 'i18n', 'migrations', 'migration',
}


def is_source_code_path(path: str) -> bool:
    """Check if path points to a source code file."""
    # Check extension
    ext = '.' + path.rsplit('.', 1)[-1].lower() if '.' in path else ''
    if ext not in SOURCE_CODE_EXTENSIONS:
        return False
    
    # Check for excluded directories
    parts = path.lower().split('/')
    for part in parts[:-1]:  # Check directory parts, not filename
        if part in LEXICAL_EXCLUDE_DIRS:
            return False
    
    return True


def extract_path_stems(path: str) -> Set[str]:
    """Extract meaningful stems from a file path for matching."""
    # Get filename without extension and directory parts
    parts = path.lower().replace('/', ' ').replace('_', ' ').replace('-', ' ')
    # Remove extension
    if '.' in parts:
        parts = parts.rsplit('.', 1)[0]
    # Extract words of 3+ chars
    return {w for w in re.findall(r'[a-z]+', parts) if len(w) >= 3}


def find_path_matches(query: str, file_paths: List[str], source_only: bool = True) -> List[Tuple[str, str]]:
    """
    Find files where query terms match the path bidirectionally.
    
    Matches when:
    - Query term appears in path (e.g., "auth" in "auth.py")
    - Path stem appears in query term (e.g., "auth" from path in "authentication")
    
    Args:
        query: User query string
        file_paths: List of file paths to search
        source_only: If True, only return source code files (filter out .json, .md, results/)
    
    Returns:
        List of (path, matched_term) tuples, with source files first.
    """
    terms = extract_query_terms(query)
    source_matches = []
    other_matches = []
    
    for path in file_paths:
        path_lower = path.lower()
        path_stems = extract_path_stems(path)
        
        for term in terms:
            matched = False
            match_reason = term
            
            # Forward match: query term in path (e.g., "auth" in "auth.py")
            if term in path_lower:
                matched = True
            # Stem match with -ing suffix (e.g., "binding" matches "bindings")
            elif term.endswith('ing') and term[:-3] in path_lower:
                matched = True
                match_reason = term[:-3]
            else:
                # Reverse match: path stem in query term (e.g., "auth" in "authentication")
                for stem in path_stems:
                    if len(stem) >= 4 and stem in term:  # Require 4+ char stems to avoid false positives
                        matched = True
                        match_reason = f"{stem}→{term}"
                        break
            
            if matched:
                is_source = is_source_code_path(path)
                
                if source_only and not is_source:
                    # Skip non-source files when source_only is True
                    break
                
                if is_source:
                    source_matches.append((path, match_reason))
                else:
                    other_matches.append((path, match_reason))
                break
    
    # Return source matches first, then other matches
    return source_matches + other_matches


# =============================================================================
# Pricing (USD per 1M tokens) - Update as needed
# =============================================================================

LLM_PRICING = {
    # OpenAI
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # Anthropic
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # Google Gemini
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
}

# Default pricing if model not in list
DEFAULT_PRICING = {"input": 0.10, "output": 0.40}


# =============================================================================
# System prompt for file selection
# =============================================================================

SYSTEM_PROMPT = """You are an expert software architect. Your job is to pick the most relevant source files from a codebase, given a user question.

Prioritize (in this order):
1. CORE SOURCE CODE in lib/, src/, packages/*/src/, or similar directories - this is most important!
2. Entry points (main.py, index.ts, app.py, server.js, index.js, but only at root or in src/)
3. Files with names matching keywords in the question
4. Configuration files if asking about setup/config
5. Internal implementation files that define the core logic

For MONOREPOS with packages/ or apps/ directories:
- Look INSIDE subdirectories for actual source (e.g., packages/react-dom/src/, apps/web/src/)
- The top-level packages/*/index.js is often just a re-export; dig deeper for implementation

STRONGLY AVOID (unless explicitly asked):
- Files in test/, tests/, __tests__/, spec/, or similar test directories
- Files in examples/, example/, demo/, demos/ directories  
- Files in docs/, documentation/ directories
- Test files (*.test.js, *.spec.ts, *_test.py, etc.)
- Mock files or fixtures

Do NOT include:
- Generated files, minified files, or lock files
- Binary files or assets
- Files unrelated to the question

When answering questions about how something works, focus on the SOURCE IMPLEMENTATION, not examples or tests. Only return file paths that are likely needed to answer the question. Be selective - quality over quantity."""


def build_user_prompt(file_tree_text: str, query: str, max_files: int = 10, path_matches: List[Tuple[str, str]] = None) -> str:
    """Build the user prompt for the selector LLM."""
    
    # Add path matches hint if available
    path_hint = ""
    if path_matches:
        match_list = "\n".join(f"  - {path} (matches '{term}')" for path, term in path_matches[:10])
        path_hint = f"""
LEXICAL HINTS (files with query terms in path - consider these carefully):
{match_list}

"""
    
    return f"""FILE_TREE:
{file_tree_text}
{path_hint}
QUESTION:
"{query}"

Return a JSON array of up to {max_files} file paths from the tree above that are most relevant to answering the question.
The paths must exactly match entries in the file tree (case-sensitive).

Return ONLY the JSON array, no explanation. Example format:
["src/auth/login.ts", "src/routes/userRoutes.ts"]"""


def build_user_prompt_with_reasons(file_tree_text: str, query: str, max_files: int = 10, path_matches: List[Tuple[str, str]] = None) -> str:
    """Build the user prompt for the selector LLM, requesting reasons and truncation priorities."""
    
    # Add path matches hint if available
    path_hint = ""
    if path_matches:
        match_list = "\n".join(f"  - {path} (matches '{term}')" for path, term in path_matches[:10])
        path_hint = f"""
LEXICAL HINTS (files with query terms in path - consider these carefully):
{match_list}

"""
    
    return f"""FILE_TREE:
{file_tree_text}
{path_hint}
QUESTION:
"{query}"

Return a JSON array of up to {max_files} objects, each with "path", "reason", and "priority" fields.
The paths must exactly match entries in the file tree (case-sensitive).
The reason should be a brief (5-15 words) explanation of why this file is relevant.

Priority levels (how much content is needed from this file):
- "critical" - Essential for answering query, needs substantial content (40-60% of budget)
- "important" - Key supporting file, needs meaningful chunks (15-25% of budget)
- "supplementary" - Helpful context, can be heavily truncated (5-10% of budget)
- "reference" - Just need to know it exists, minimal content (1-5% of budget)

Consider:
- Deep dive queries ("how does X work in detail?") → 1-2 critical files with 50-60% each
- Multi-system queries ("X and Y") → Split critical budget between systems
- Broad queries ("overall architecture") → More files at 'important', less per file
- Specific feature queries → The exact feature file gets 'critical', not just the first file

Return ONLY the JSON array, no explanation. Example format:
[
  {{"path": "src/auth/login.ts", "reason": "handles user authentication flow", "priority": "critical"}},
  {{"path": "src/routes/userRoutes.ts", "reason": "defines API endpoints", "priority": "important"}},
  {{"path": "src/types/user.ts", "reason": "user type definitions", "priority": "reference"}}
]"""


# =============================================================================
# LLM API calls
# =============================================================================

async def call_openai(
    messages: List[Dict],
    model: str,
    timeout: float = 30.0
) -> Tuple[str, int, int]:
    """
    Call OpenAI API.
    
    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 1024,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        return content, input_tokens, output_tokens


async def call_anthropic(
    messages: List[Dict],
    model: str,
    timeout: float = 30.0
) -> Tuple[str, int, int]:
    """
    Call Anthropic API.
    
    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    """
    # Convert from OpenAI format to Anthropic format
    # Anthropic wants system as separate param, not in messages
    system_content = ""
    user_messages = []
    
    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        else:
            user_messages.append(msg)
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "system": system_content,
                "messages": user_messages,
                "max_tokens": 1024,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["content"][0]["text"]
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        return content, input_tokens, output_tokens


async def call_gemini(
    messages: List[Dict],
    model: str,
    timeout: float = 30.0
) -> Tuple[str, int, int]:
    """
    Call Google Gemini API.
    
    Returns:
        Tuple of (response_text, input_tokens, output_tokens)
    """
    # Convert from OpenAI format to Gemini format
    # Combine system + user messages into a single prompt
    combined_text = ""
    for msg in messages:
        if msg["role"] == "system":
            combined_text += msg["content"] + "\n\n"
        elif msg["role"] == "user":
            combined_text += msg["content"]
    
    # Gemini uses the model name in the URL
    # Strip "models/" prefix if present
    model_name = model.replace("models/", "")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            url,
            params={"key": settings.LLM_API_KEY},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [{"text": combined_text}]
                }],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 4096,
                }
            },
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        
        # Extract content from Gemini response
        content = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # DEBUG: Log raw response
        print(f"[DEBUG] Gemini raw response: {content[:500]}")
        
        # Extract token usage
        usage = data.get("usageMetadata", {})
        input_tokens = usage.get("promptTokenCount", 0)
        output_tokens = usage.get("candidatesTokenCount", 0)
        
        return content, input_tokens, output_tokens


# =============================================================================
# Response parsing
# =============================================================================

def parse_file_paths(response_text: str) -> List[str]:
    """
    Parse file paths from LLM response.
    
    Handles:
    - Clean JSON array
    - JSON wrapped in markdown code blocks
    - Malformed responses
    """
    text = response_text.strip()
    
    # Try to extract JSON from markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if code_block_match:
        text = code_block_match.group(1).strip()
    
    # Try to find array in the text
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        text = array_match.group(0)
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            # Filter to only strings
            return [p for p in parsed if isinstance(p, str)]
    except json.JSONDecodeError:
        pass
    
    return []


def parse_file_paths_with_reasons(response_text: str) -> List[Tuple[str, dict]]:
    """
    Parse file paths with reasons and priorities from LLM response.
    
    Returns:
        List of (path, metadata_dict) tuples where metadata_dict contains reason and priority
    """
    text = response_text.strip()
    
    # Try to extract JSON from markdown code blocks (complete block)
    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if code_block_match:
        text = code_block_match.group(1).strip()
    else:
        # Handle truncated code block (no closing ```) - grab everything after opening
        partial_block_match = re.search(r'```(?:json)?\s*([\s\S]*)', text)
        if partial_block_match:
            text = partial_block_match.group(1).strip()
    
    # Try to find array in the text (complete)
    array_match = re.search(r'\[[\s\S]*\]', text)
    if array_match:
        text = array_match.group(0)
    else:
        # Handle truncated array — find opening [ and try to close it
        array_start = text.find('[')
        if array_start != -1:
            text = text[array_start:]
            # Try to close the array: first close any open object, then close array
            if not text.rstrip().endswith(']'):
                # Drop the last incomplete line and close
                lines = text.rstrip().rstrip(',').rsplit('\n', 1)[0]
                # If we're inside an object, close it
                if lines.rstrip().endswith('"') or lines.rstrip()[-1] not in (']', '}'):
                    lines += '"}'
                elif not lines.rstrip().endswith('}'):
                    lines += '}'
                text = lines + ']'
    
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                if isinstance(item, dict) and "path" in item:
                    path = item["path"]
                    reason = item.get("reason", "")
                    priority = item.get("priority", "important")  # Default to important
                    if isinstance(path, str):
                        # Return dict with both reason and priority
                        results.append((path, {"reason": reason, "priority": priority}))
                elif isinstance(item, str):
                    # Fallback: plain string without reason
                    results.append((item, {"reason": "", "priority": "important"}))
            return results
    except json.JSONDecodeError:
        pass
    
    return []


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str
) -> float:
    """Calculate cost in USD for an LLM call."""
    pricing = LLM_PRICING.get(model, DEFAULT_PRICING)
    
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


# =============================================================================
# Main selector function
# =============================================================================

async def call_selector_llm(
    file_tree_text: str,
    query: str,
    max_files: int = 10,
    include_reasons: bool = False,
    file_paths: List[str] = None,
) -> Tuple[List[str], float, int, dict]:
    """
    Call LLM to select relevant files from the file tree.
    
    Args:
        file_tree_text: Text representation of the file tree
        query: User's question about the codebase
        max_files: Maximum number of files to select
        include_reasons: If True, ask LLM for reason per file selection
        file_paths: List of all file paths (for lexical matching)
    
    Returns:
        Tuple of (selected_paths, cost_usd, elapsed_ms, reasons_dict)
        reasons_dict maps path -> reason string (empty if include_reasons=False)
    """
    start_time = time.perf_counter()
    
    # Find lexical path matches
    path_matches = None
    if file_paths:
        path_matches = find_path_matches(query, file_paths)
        if path_matches:
            print(f"[LEXICAL] Found {len(path_matches)} path matches: {[p for p, _ in path_matches[:5]]}")
    
    # Build messages - use different prompt if reasons requested
    if include_reasons:
        user_prompt = build_user_prompt_with_reasons(file_tree_text, query, max_files, path_matches)
    else:
        user_prompt = build_user_prompt(file_tree_text, query, max_files, path_matches)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    
    model = settings.LLM_MODEL
    provider = settings.LLM_PROVIDER.lower()
    reasons_dict = {}
    
    try:
        if provider == "gemini" or provider == "google":
            response_text, input_tokens, output_tokens = await call_gemini(
                messages, model
            )
        elif provider == "anthropic":
            response_text, input_tokens, output_tokens = await call_anthropic(
                messages, model
            )
        else:
            # Default to OpenAI
            response_text, input_tokens, output_tokens = await call_openai(
                messages, model
            )
        
        # Parse response
        if include_reasons:
            parsed = parse_file_paths_with_reasons(response_text)
            selected_paths = [p for p, _ in parsed]
            reasons_dict = {p: meta for p, meta in parsed}  # meta is already a dict with reason+priority
        else:
            selected_paths = parse_file_paths(response_text)
        
        # Calculate cost
        cost_usd = calculate_cost(input_tokens, output_tokens, model)
        
    except Exception as e:
        # On error, return empty selection with zero cost
        # Orchestrator will use fallback
        print(f"Selector LLM error: {e}")
        selected_paths = []
        cost_usd = 0.0
    
    elapsed_ms = int((time.perf_counter() - start_time) * 1000)
    
    return selected_paths, cost_usd, elapsed_ms, reasons_dict


# =============================================================================
# Fallback heuristic selector
# =============================================================================

def fallback_select(
    file_index: List[Dict],
    query: str,
    max_files: int = 5
) -> List[str]:
    """
    Fallback file selection using heuristics when LLM fails.
    
    Priority:
    1. README files
    2. Entry points (main.*, index.*, app.*, server.*)
    3. Config files
    4. Files matching query keywords
    5. Shallow files (depth 1-2)
    """
    if not file_index:
        return []
    
    selected: List[str] = []
    query_lower = query.lower()
    
    # Extract keywords from query (simple tokenization)
    keywords = set(re.findall(r'\b[a-z]{3,}\b', query_lower))
    # Remove common words
    keywords -= {"the", "how", "does", "what", "where", "why", "can", "this", "that", "from", "with"}
    
    # Score each file
    scored_files: List[Tuple[float, str]] = []
    
    for entry in file_index:
        path = entry["path"]
        path_lower = path.lower()
        name = path.split("/")[-1].lower()
        depth = entry["depth"]
        score = 0.0
        
        # README files get high priority
        if "readme" in name:
            score += 100
        
        # Entry points
        entry_patterns = ["main.", "index.", "app.", "server.", "cli.", "__main__"]
        if any(pattern in name for pattern in entry_patterns):
            score += 50
        
        # Config files
        config_patterns = ["config", "settings", ".env", "package.json", "pyproject.toml", 
                          "setup.py", "cargo.toml", "go.mod", "requirements.txt"]
        if any(pattern in name for pattern in config_patterns):
            score += 30
        
        # Keyword matching in path
        for keyword in keywords:
            if keyword in path_lower:
                score += 20
        
        # Prefer shallow files
        if depth == 1:
            score += 10
        elif depth == 2:
            score += 5
        
        # Prefer source files
        source_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".go", ".rs", ".rb", ".java"}
        if entry["ext"] in source_exts:
            score += 5
        
        scored_files.append((score, path))
    
    # Sort by score (descending) and take top N
    scored_files.sort(key=lambda x: (-x[0], x[1]))
    
    return [path for score, path in scored_files[:max_files]]
