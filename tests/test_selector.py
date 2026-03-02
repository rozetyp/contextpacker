"""
Unit tests for selector module (selector.py).

Tests lexical path matching and query term extraction.
These determine which files get "hints" before LLM selection.
"""

import pytest


class TestExtractQueryTerms:
    """Tests for extracting meaningful terms from queries."""
    
    def test_extracts_words(self):
        """Should extract words from query."""
        from context_packer.selector import extract_query_terms
        
        terms = extract_query_terms("How does authentication work?")
        
        assert "authentication" in terms
    
    def test_filters_short_words(self):
        """Should filter words shorter than 3 characters."""
        from context_packer.selector import extract_query_terms
        
        terms = extract_query_terms("How do I fix it?")
        
        assert "do" not in terms
        assert "it" not in terms
        assert "fix" in terms
    
    def test_filters_stopwords(self):
        """Should filter common stopwords."""
        from context_packer.selector import extract_query_terms
        
        terms = extract_query_terms("How does the routing work with this framework?")
        
        assert "how" not in terms
        assert "does" not in terms
        assert "the" not in terms
        assert "with" not in terms
        assert "work" not in terms  # "work/works" are stopwords
        assert "routing" in terms
        assert "framework" in terms
    
    def test_filters_repo_names(self):
        """Should filter common repo names."""
        from context_packer.selector import extract_query_terms
        
        terms = extract_query_terms("How does Flask handle routing?")
        
        assert "flask" not in terms
        assert "routing" in terms
        assert "handle" in terms
    
    def test_lowercase_output(self):
        """Should return lowercase terms."""
        from context_packer.selector import extract_query_terms
        
        terms = extract_query_terms("Where is the AuthController?")
        
        # All terms should be lowercase
        for term in terms:
            assert term == term.lower()


class TestIsSourceCodePath:
    """Tests for determining if a path is source code."""
    
    def test_python_is_source(self):
        """Python files are source code."""
        from context_packer.selector import is_source_code_path
        
        assert is_source_code_path("src/main.py")
        assert is_source_code_path("lib/utils.py")
    
    def test_javascript_is_source(self):
        """JavaScript files are source code."""
        from context_packer.selector import is_source_code_path
        
        assert is_source_code_path("src/app.js")
        assert is_source_code_path("lib/router.jsx")
    
    def test_typescript_is_source(self):
        """TypeScript files are source code."""
        from context_packer.selector import is_source_code_path
        
        assert is_source_code_path("src/index.ts")
        assert is_source_code_path("components/Button.tsx")
    
    def test_markdown_not_source(self):
        """Markdown files are not source code."""
        from context_packer.selector import is_source_code_path
        
        assert not is_source_code_path("README.md")
        assert not is_source_code_path("docs/guide.md")
    
    def test_json_not_source(self):
        """JSON files are not source code."""
        from context_packer.selector import is_source_code_path
        
        assert not is_source_code_path("package.json")
        assert not is_source_code_path("config/settings.json")
    
    def test_excludes_results_dir(self):
        """Files in results/ are not source code."""
        from context_packer.selector import is_source_code_path
        
        assert not is_source_code_path("results/output.py")
        assert not is_source_code_path("output/report.js")
    
    def test_excludes_docs_dir(self):
        """Files in docs/ are not source code."""
        from context_packer.selector import is_source_code_path
        
        assert not is_source_code_path("docs/example.py")
        assert not is_source_code_path("documentation/sample.js")
    
    def test_excludes_examples_dir(self):
        """Files in examples/ are not source code."""
        from context_packer.selector import is_source_code_path
        
        assert not is_source_code_path("examples/demo.py")
        assert not is_source_code_path("example/tutorial.js")


class TestExtractPathStems:
    """Tests for extracting searchable stems from file paths."""
    
    def test_extracts_filename(self):
        """Should extract words from filename."""
        from context_packer.selector import extract_path_stems
        
        stems = extract_path_stems("src/auth_controller.py")
        
        assert "auth" in stems
        assert "controller" in stems
    
    def test_extracts_directory_names(self):
        """Should extract words from directory names."""
        from context_packer.selector import extract_path_stems
        
        stems = extract_path_stems("authentication/handlers/login.py")
        
        assert "authentication" in stems
        assert "handlers" in stems
        assert "login" in stems
    
    def test_splits_snake_case(self):
        """Should split snake_case names."""
        from context_packer.selector import extract_path_stems
        
        stems = extract_path_stems("user_authentication.py")
        
        assert "user" in stems
        assert "authentication" in stems
    
    def test_splits_kebab_case(self):
        """Should split kebab-case names."""
        from context_packer.selector import extract_path_stems
        
        stems = extract_path_stems("user-authentication.js")
        
        assert "user" in stems
        assert "authentication" in stems
    
    def test_filters_short_stems(self):
        """Should filter stems shorter than 3 characters."""
        from context_packer.selector import extract_path_stems
        
        stems = extract_path_stems("a_bc_def.py")
        
        assert "a" not in stems
        assert "bc" not in stems
        assert "def" in stems


class TestFindPathMatches:
    """Tests for finding files matching query terms."""
    
    def test_forward_match(self):
        """Should match when query term appears in path."""
        from context_packer.selector import find_path_matches
        
        paths = ["src/auth.py", "src/models.py", "lib/router.js"]
        
        matches = find_path_matches("How does auth work?", paths)
        
        assert any("auth.py" in m[0] for m in matches)
    
    def test_reverse_match(self):
        """Should match when path stem appears in query term."""
        from context_packer.selector import find_path_matches
        
        paths = ["src/auth.py", "src/models.py"]
        
        # "authentication" contains "auth"
        matches = find_path_matches("How does authentication work?", paths)
        
        assert any("auth.py" in m[0] for m in matches)
    
    def test_ing_suffix_match(self):
        """Should match -ing suffix variations."""
        from context_packer.selector import find_path_matches
        
        paths = ["bindings/__init__.py", "models.py"]
        
        # "binding" should match "bindings"
        matches = find_path_matches("Where is request body binding?", paths)
        
        assert any("bindings" in m[0] for m in matches)
    
    def test_source_only_mode(self):
        """Should filter non-source files in source_only mode."""
        from context_packer.selector import find_path_matches
        
        paths = ["src/auth.py", "docs/auth.md", "results/auth.json"]
        
        matches = find_path_matches("auth", paths, source_only=True)
        
        # Only .py should match
        paths_matched = [m[0] for m in matches]
        assert "src/auth.py" in paths_matched
        assert "docs/auth.md" not in paths_matched
        assert "results/auth.json" not in paths_matched
    
    def test_source_files_first(self):
        """Source files should appear before non-source in results."""
        from context_packer.selector import find_path_matches
        
        paths = ["docs/auth.md", "src/auth.py", "config/auth.json"]
        
        matches = find_path_matches("auth", paths, source_only=False)
        
        # If both types present, source should be first
        if len(matches) >= 2:
            source_positions = [i for i, (p, _) in enumerate(matches) if p.endswith(".py")]
            non_source_positions = [i for i, (p, _) in enumerate(matches) if not p.endswith(".py")]
            
            if source_positions and non_source_positions:
                assert min(source_positions) < min(non_source_positions)
    
    def test_returns_match_reason(self):
        """Should return the matched term as reason."""
        from context_packer.selector import find_path_matches
        
        paths = ["src/authentication.py"]
        
        matches = find_path_matches("auth handler", paths)
        
        if matches:
            path, reason = matches[0]
            assert reason is not None
    
    def test_empty_paths(self):
        """Should handle empty paths list."""
        from context_packer.selector import find_path_matches
        
        matches = find_path_matches("auth", [])
        
        assert matches == []
    
    def test_no_matches(self):
        """Should return empty list when no matches."""
        from context_packer.selector import find_path_matches
        
        paths = ["src/database.py", "lib/cache.js"]
        
        matches = find_path_matches("authentication routing", paths)
        
        assert matches == []


class TestBuildUserPrompt:
    """Tests for building the LLM user prompt."""
    
    def test_includes_file_tree(self):
        """Should include file tree in prompt."""
        from context_packer.selector import build_user_prompt
        
        tree = "src/\n  main.py\n  utils.py"
        query = "How does the main app work?"
        
        prompt = build_user_prompt(tree, query, max_files=10)
        
        assert "main.py" in prompt
        assert "utils.py" in prompt
    
    def test_includes_query(self):
        """Should include user query in prompt."""
        from context_packer.selector import build_user_prompt
        
        prompt = build_user_prompt("src/", "How does routing work?", max_files=10)
        
        assert "How does routing work?" in prompt
    
    def test_includes_max_files(self):
        """Should include max files limit."""
        from context_packer.selector import build_user_prompt
        
        prompt = build_user_prompt("src/", "query", max_files=5)
        
        assert "5" in prompt
    
    def test_includes_path_matches_hint(self):
        """Should include lexical hints when provided."""
        from context_packer.selector import build_user_prompt
        
        path_matches = [("src/auth.py", "auth"), ("lib/login.js", "login")]
        
        prompt = build_user_prompt("src/", "query", max_files=10, path_matches=path_matches)
        
        assert "LEXICAL HINTS" in prompt
        assert "src/auth.py" in prompt


class TestFallbackSelect:
    """Tests for fallback file selection when LLM fails."""
    
    def test_selects_readme(self):
        """Should prioritize README files."""
        from context_packer.selector import fallback_select
        
        file_index = [
            {"path": "src/main.py", "depth": 2, "ext": ".py"},
            {"path": "README.md", "depth": 1, "ext": ".md"},
            {"path": "lib/utils.js", "depth": 2, "ext": ".js"},
        ]
        
        selected = fallback_select(file_index, "How does this work?", max_files=5)
        
        # README should be included
        assert "README.md" in selected
    
    def test_selects_entry_points(self):
        """Should prioritize main/index files."""
        from context_packer.selector import fallback_select
        
        file_index = [
            {"path": "src/helper.py", "depth": 2, "ext": ".py"},
            {"path": "main.py", "depth": 1, "ext": ".py"},
            {"path": "docs/guide.md", "depth": 2, "ext": ".md"},
        ]
        
        selected = fallback_select(file_index, "How does this work?", max_files=5)
        
        assert "main.py" in selected
    
    def test_respects_max_files(self):
        """Should respect max files limit."""
        from context_packer.selector import fallback_select
        
        file_index = [{"path": f"file_{i}.py", "depth": 1, "ext": ".py"} for i in range(20)]
        
        selected = fallback_select(file_index, "query", max_files=3)
        
        assert len(selected) <= 3
    
    def test_keyword_matching(self):
        """Should boost files matching query keywords."""
        from context_packer.selector import fallback_select
        
        file_index = [
            {"path": "src/database.py", "depth": 2, "ext": ".py"},
            {"path": "src/authentication.py", "depth": 2, "ext": ".py"},
            {"path": "src/logging.py", "depth": 2, "ext": ".py"},
        ]
        
        selected = fallback_select(file_index, "How does authentication work?", max_files=5)
        
        # Auth file should be included due to keyword match
        assert "src/authentication.py" in selected
