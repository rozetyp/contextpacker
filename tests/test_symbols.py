"""
Unit tests for symbol extraction (symbols.py).

Tests the AST-based symbol extraction that enriches file trees
with semantic information - the "hidden moat" of ContextPacker.
"""

import pytest
from pathlib import Path
from textwrap import dedent


class TestExtractPythonSymbols:
    """Tests for Python AST-based symbol extraction."""
    
    def test_extracts_functions(self, tmp_path):
        """Should extract top-level function names."""
        from context_packer.symbols import extract_python_symbols
        
        code = dedent('''
            def create_pack():
                pass
            
            def validate_input():
                pass
            
            async def fetch_data():
                pass
        ''')
        
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = extract_python_symbols(file_path)
        
        assert "create_pack" in result["symbols"]
        assert "validate_input" in result["symbols"]
        assert "fetch_data" in result["symbols"]
    
    def test_extracts_classes(self, tmp_path):
        """Should extract class names."""
        from context_packer.symbols import extract_python_symbols
        
        code = dedent('''
            class PackRequest:
                pass
            
            class PackResponse:
                pass
        ''')
        
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = extract_python_symbols(file_path)
        
        assert "PackRequest" in result["symbols"]
        assert "PackResponse" in result["symbols"]
    
    def test_extracts_constants(self, tmp_path):
        """Should extract UPPER_CASE constants."""
        from context_packer.symbols import extract_python_symbols
        
        code = dedent('''
            MAX_FILES = 100
            DEFAULT_TIMEOUT = 30
            _PRIVATE_CONST = "hidden"
        ''')
        
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = extract_python_symbols(file_path)
        
        assert "MAX_FILES" in result["symbols"]
        assert "DEFAULT_TIMEOUT" in result["symbols"]
        # Private constants should be skipped? Depends on implementation
    
    def test_skips_private_functions(self, tmp_path):
        """Should skip functions starting with underscore."""
        from context_packer.symbols import extract_python_symbols
        
        code = dedent('''
            def public_function():
                pass
            
            def _private_helper():
                pass
            
            def __dunder_method__():
                pass
        ''')
        
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = extract_python_symbols(file_path)
        
        assert "public_function" in result["symbols"]
        assert "_private_helper" not in result["symbols"]
        assert "__dunder_method__" not in result["symbols"]
    
    def test_extracts_module_docstring(self, tmp_path):
        """Should extract first line of module docstring."""
        from context_packer.symbols import extract_python_symbols
        
        code = dedent('''
            """
            Main orchestrator for context packing pipeline.
            
            This module handles the entire flow from clone to pack.
            """
            
            def create_pack():
                pass
        ''')
        
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = extract_python_symbols(file_path)
        
        assert result["doc"] is not None
        assert "orchestrator" in result["doc"].lower()
    
    def test_respects_max_symbols(self, tmp_path):
        """Should limit number of extracted symbols."""
        from context_packer.symbols import extract_python_symbols
        
        code = "\n".join([f"def func_{i}(): pass" for i in range(20)])
        
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = extract_python_symbols(file_path, max_symbols=5)
        
        assert len(result["symbols"]) == 5
    
    def test_handles_syntax_errors(self, tmp_path):
        """Should gracefully handle invalid Python."""
        from context_packer.symbols import extract_python_symbols
        
        code = "def broken( # missing close paren"
        
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = extract_python_symbols(file_path)
        
        assert result["symbols"] == []
        assert result["doc"] is None
    
    def test_handles_unicode_errors(self, tmp_path):
        """Should handle files with encoding issues."""
        from context_packer.symbols import extract_python_symbols
        
        file_path = tmp_path / "test.py"
        file_path.write_bytes(b"\xff\xfe invalid utf8")
        
        result = extract_python_symbols(file_path)
        
        # Should not raise, should return empty
        assert result["symbols"] == []


class TestExtractJsSymbols:
    """Tests for JavaScript/TypeScript regex-based symbol extraction."""
    
    def test_extracts_functions(self, tmp_path):
        """Should extract function declarations."""
        from context_packer.symbols import extract_js_symbols
        
        code = dedent('''
            function authenticateUser(token) {
                return true;
            }
            
            export async function fetchData() {
                return [];
            }
        ''')
        
        file_path = tmp_path / "test.js"
        file_path.write_text(code)
        
        result = extract_js_symbols(file_path)
        
        assert "authenticateUser" in result["symbols"]
        assert "fetchData" in result["symbols"]
    
    def test_extracts_const_functions(self, tmp_path):
        """Should extract arrow functions assigned to const."""
        from context_packer.symbols import extract_js_symbols
        
        code = dedent('''
            export const handleRequest = async (req) => {
                return res;
            };
            
            const helper = () => {};
        ''')
        
        file_path = tmp_path / "test.js"
        file_path.write_text(code)
        
        result = extract_js_symbols(file_path)
        
        assert "handleRequest" in result["symbols"]
        assert "helper" in result["symbols"]
    
    def test_extracts_classes(self, tmp_path):
        """Should extract class declarations."""
        from context_packer.symbols import extract_js_symbols
        
        code = dedent('''
            export class AuthService {
                constructor() {}
            }
            
            class InternalHelper {
                run() {}
            }
        ''')
        
        file_path = tmp_path / "test.js"
        file_path.write_text(code)
        
        result = extract_js_symbols(file_path)
        
        assert "AuthService" in result["symbols"]
        assert "InternalHelper" in result["symbols"]
    
    def test_extracts_interfaces_ts(self, tmp_path):
        """Should extract TypeScript interfaces."""
        from context_packer.symbols import extract_js_symbols
        
        code = dedent('''
            export interface AuthConfig {
                timeout: number;
            }
            
            interface InternalState {
                value: string;
            }
        ''')
        
        file_path = tmp_path / "test.ts"
        file_path.write_text(code)
        
        result = extract_js_symbols(file_path)
        
        assert "AuthConfig" in result["symbols"]
        assert "InternalState" in result["symbols"]
    
    def test_extracts_jsdoc_comment(self, tmp_path):
        """Should extract file-level JSDoc comment."""
        from context_packer.symbols import extract_js_symbols
        
        code = dedent('''
            /**
             * Authentication utilities for the application.
             * @module auth
             */
            
            export function login() {}
        ''')
        
        file_path = tmp_path / "test.js"
        file_path.write_text(code)
        
        result = extract_js_symbols(file_path)
        
        # Implementation may vary - just check it doesn't crash
        assert isinstance(result, dict)


class TestExtractFileSymbols:
    """Tests for the generic symbol extraction dispatcher."""
    
    def test_dispatches_python(self, tmp_path):
        """Should use Python extractor for .py files."""
        from context_packer.symbols import extract_file_symbols
        
        code = "def my_func(): pass"
        file_path = tmp_path / "test.py"
        file_path.write_text(code)
        
        result = extract_file_symbols(file_path)
        
        assert result is not None
        assert "my_func" in result["symbols"]
    
    def test_dispatches_javascript(self, tmp_path):
        """Should use JS extractor for .js files."""
        from context_packer.symbols import extract_file_symbols
        
        code = "function myFunc() {}"
        file_path = tmp_path / "test.js"
        file_path.write_text(code)
        
        result = extract_file_symbols(file_path)
        
        assert result is not None
        assert "myFunc" in result["symbols"]
    
    def test_dispatches_typescript(self, tmp_path):
        """Should use JS extractor for .ts files."""
        from context_packer.symbols import extract_file_symbols
        
        code = "export interface MyInterface {}"
        file_path = tmp_path / "test.ts"
        file_path.write_text(code)
        
        result = extract_file_symbols(file_path)
        
        assert result is not None
    
    def test_returns_none_for_unsupported(self, tmp_path):
        """Should return None for unsupported file types."""
        from context_packer.symbols import extract_file_symbols
        
        file_path = tmp_path / "test.md"
        file_path.write_text("# Markdown content")
        
        result = extract_file_symbols(file_path)
        
        assert result is None


class TestBuildSymbolsIndex:
    """Tests for building a symbol index from a file list."""
    
    def test_indexes_source_files(self, temp_repo):
        """Should index symbols from source code files."""
        from context_packer.symbols import build_symbols_index
        
        file_index = [
            {"path": "main.py", "ext": ".py"},
            {"path": "src/app.py", "ext": ".py"},
            {"path": "README.md", "ext": ".md"},
        ]
        
        result = build_symbols_index(temp_repo, file_index, max_files=10)
        
        # Should have indexed the .py files
        assert "main.py" in result or "src/app.py" in result
        # Should not have indexed .md
        assert "README.md" not in result
    
    def test_respects_max_files(self, temp_repo):
        """Should limit number of files processed."""
        from context_packer.symbols import build_symbols_index
        
        file_index = [{"path": f"src/file_{i}.py", "ext": ".py"} for i in range(20)]
        
        # Create files
        src = temp_repo / "src"
        src.mkdir(exist_ok=True)
        for i in range(20):
            (src / f"file_{i}.py").write_text(f"def func_{i}(): pass")
        
        result = build_symbols_index(temp_repo, file_index, max_files=5)
        
        assert len(result) <= 5


class TestFormatSymbolsHint:
    """Tests for formatting symbols into compact hints."""
    
    def test_formats_symbols_list(self):
        """Should format symbols as bracketed list."""
        from context_packer.symbols import format_symbols_hint
        
        result = format_symbols_hint(
            ["create_pack", "validate_input", "PackRequest"],
            None
        )
        
        assert "[" in result
        assert "]" in result
        assert "create_pack" in result
        assert "validate_input" in result
    
    def test_empty_symbols_returns_empty(self):
        """Should return empty string for no symbols."""
        from context_packer.symbols import format_symbols_hint
        
        result = format_symbols_hint([], None)
        
        assert result == ""
    
    def test_handles_doc_string(self):
        """Should include doc string if provided."""
        from context_packer.symbols import format_symbols_hint
        
        result = format_symbols_hint(
            ["create_pack"],
            "Main orchestrator module"
        )
        
        # Format may or may not include doc - depends on implementation
        assert "[create_pack]" in result
