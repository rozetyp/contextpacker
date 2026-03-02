"""
Symbol extraction for enriching file tree with content metadata.

Extracts top-level symbols (functions, classes) and docstrings from source files
to give the LLM selector better information about file contents.

This addresses the "name-based semantic guesser" limitation where the LLM
can only guess file contents from filenames.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Python Symbol Extraction (AST-based)
# =============================================================================

def extract_python_symbols(file_path: Path, max_symbols: int = 20) -> Dict:
    """
    Extract top-level symbols and docstring from a Python file.
    
    Args:
        file_path: Path to Python file
        max_symbols: Max number of symbols to extract
        
    Returns:
        Dict with keys:
            - symbols: List of top-level function/class names
            - doc: First line of module docstring (or None)
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
    except (SyntaxError, UnicodeDecodeError, OSError):
        return {"symbols": [], "doc": None}
    
    symbols = []
    doc = None
    
    # Get module docstring
    if (tree.body and 
        isinstance(tree.body[0], ast.Expr) and 
        isinstance(tree.body[0].value, ast.Constant) and
        isinstance(tree.body[0].value.value, str)):
        full_doc = tree.body[0].value.value.strip()
        # Take first line only
        doc = full_doc.split('\n')[0].strip()
        if len(doc) > 80:
            doc = doc[:77] + "..."
    
    # Extract top-level symbols
    for node in tree.body:
        if len(symbols) >= max_symbols:
            break
            
        if isinstance(node, ast.FunctionDef):
            # Skip private functions (leading underscore)
            if not node.name.startswith('_'):
                symbols.append(node.name)
        elif isinstance(node, ast.AsyncFunctionDef):
            if not node.name.startswith('_'):
                symbols.append(node.name)
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith('_'):
                symbols.append(node.name)
        elif isinstance(node, ast.Assign):
            # Capture important constants (UPPER_CASE)
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    if len(symbols) < max_symbols:
                        symbols.append(target.id)
    
    return {"symbols": symbols, "doc": doc}


# =============================================================================
# JavaScript/TypeScript Symbol Extraction (Regex-based)
# =============================================================================

# Patterns for JS/TS symbol extraction
JS_FUNCTION_PATTERN = re.compile(
    r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)',
    re.MULTILINE
)
JS_CONST_FUNCTION_PATTERN = re.compile(
    r'^(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(?',
    re.MULTILINE
)
JS_CLASS_PATTERN = re.compile(
    r'^(?:export\s+)?class\s+(\w+)',
    re.MULTILINE
)
JS_INTERFACE_PATTERN = re.compile(
    r'^(?:export\s+)?interface\s+(\w+)',
    re.MULTILINE
)


def extract_js_symbols(file_path: Path, max_symbols: int = 20) -> Dict:
    """
    Extract top-level symbols from JavaScript/TypeScript file.
    
    Uses regex for simplicity (no need for full parser).
    
    Args:
        file_path: Path to JS/TS file
        max_symbols: Max number of symbols to extract
        
    Returns:
        Dict with keys:
            - symbols: List of top-level function/class names
            - doc: First JSDoc comment or None
    """
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except (OSError, UnicodeDecodeError):
        return {"symbols": [], "doc": None}
    
    symbols = []
    doc = None
    
    # Try to get file-level JSDoc (/** ... */ at start)
    jsdoc_match = re.match(r'^\s*/\*\*\s*\n?\s*\*?\s*(.+?)(?:\n|\*/)', content)
    if jsdoc_match:
        doc = jsdoc_match.group(1).strip()
        if len(doc) > 80:
            doc = doc[:77] + "..."
    
    # Extract symbols
    all_matches = []
    
    for pattern in [JS_FUNCTION_PATTERN, JS_CONST_FUNCTION_PATTERN, 
                    JS_CLASS_PATTERN, JS_INTERFACE_PATTERN]:
        for match in pattern.finditer(content):
            name = match.group(1)
            # Skip private (leading underscore) and common noise
            if not name.startswith('_') and name not in {'module', 'exports'}:
                all_matches.append((match.start(), name))
    
    # Sort by position in file, take first N
    all_matches.sort(key=lambda x: x[0])
    seen = set()
    for _, name in all_matches:
        if name not in seen and len(symbols) < max_symbols:
            symbols.append(name)
            seen.add(name)
    
    return {"symbols": symbols, "doc": doc}


# =============================================================================
# Generic Symbol Extraction
# =============================================================================

# Map extensions to extractors
EXTRACTORS = {
    ".py": extract_python_symbols,
    ".js": extract_js_symbols,
    ".jsx": extract_js_symbols,
    ".ts": extract_js_symbols,
    ".tsx": extract_js_symbols,
    ".mjs": extract_js_symbols,
}


def extract_file_symbols(file_path: Path) -> Optional[Dict]:
    """
    Extract symbols from a source file.
    
    Args:
        file_path: Path to source file
        
    Returns:
        Dict with symbols and doc, or None if not a supported file type
    """
    ext = file_path.suffix.lower()
    extractor = EXTRACTORS.get(ext)
    
    if extractor is None:
        return None
    
    return extractor(file_path)


def build_symbols_index(
    root_dir: Path, 
    file_index: List[Dict],
    max_files: int = 100
) -> Dict[str, Dict]:
    """
    Build a symbol index for source files in the repository.
    
    Only processes files that have supported extensions and are
    high enough priority to matter.
    
    Args:
        root_dir: Root directory of the cloned repository
        file_index: List of file metadata from build_file_index
        max_files: Max number of files to extract symbols from
        
    Returns:
        Dict mapping path -> {symbols: [...], doc: "..."}
    """
    symbols_index = {}
    processed = 0
    
    for entry in file_index:
        if processed >= max_files:
            break
            
        path = entry["path"]
        ext = entry.get("ext", "").lower()
        
        # Only extract from supported source files
        if ext not in EXTRACTORS:
            continue
        
        file_path = root_dir / path
        if not file_path.exists():
            continue
        
        result = extract_file_symbols(file_path)
        if result and (result["symbols"] or result["doc"]):
            symbols_index[path] = result
            processed += 1
    
    return symbols_index


# =============================================================================
# Tree Enrichment
# =============================================================================

def format_symbols_hint(symbols: List[str], doc: Optional[str], max_len: int = 120) -> str:
    """
    Format symbols and doc into a compact hint string.
    
    Examples:
        "[clone_repo, build_index, ...]"
    """
    if not symbols:
        return ""
    
    # Truncate symbol list to prevent tree bloat
    symbols_str = ", ".join(symbols)
    hint = f"[{symbols_str}]"
    
    if len(hint) > max_len:
        # Keep as many symbols as fit
        hint = f"[{symbols_str[:max_len-5]}...]"
    
    return hint
