"""
Unit tests for git utilities (git_utils.py).

Tests priority scoring, path sorting, and file indexing logic.
These are the core algorithms that determine which files appear first.
"""

import pytest
from pathlib import Path


class TestComputePathPriority:
    """Tests for path priority scoring algorithm."""
    
    def test_src_directory_high_priority(self):
        """Files in src/ should have high priority."""
        from context_packer.git_utils import compute_path_priority
        
        score = compute_path_priority("src/main.py")
        
        # src/ gets PRIORITY_HIGH_SOURCE (+10) + source extension (+3)
        assert score >= 10
    
    def test_lib_directory_high_priority(self):
        """Files in lib/ should have high priority."""
        from context_packer.git_utils import compute_path_priority
        
        score = compute_path_priority("lib/utils.js")
        
        assert score >= 10
    
    def test_packages_src_high_priority(self):
        """Files in packages/*/src/ (monorepo) should have high priority."""
        from context_packer.git_utils import compute_path_priority
        
        score = compute_path_priority("packages/core/src/index.ts")
        
        assert score >= 10
    
    def test_root_entry_point_boost(self):
        """Root-level source files should get a boost."""
        from context_packer.git_utils import compute_path_priority
        
        root_score = compute_path_priority("main.py")
        nested_score = compute_path_priority("some/deep/path/main.py")
        
        # Root gets PRIORITY_ROOT_ENTRY (+5)
        assert root_score > nested_score
    
    def test_tests_directory_low_priority(self):
        """Files in tests/ should have low priority."""
        from context_packer.git_utils import compute_path_priority
        
        test_score = compute_path_priority("tests/test_main.py")
        src_score = compute_path_priority("src/main.py")
        
        assert test_score < src_score
    
    def test_docs_directory_low_priority(self):
        """Files in docs/ should have low priority."""
        from context_packer.git_utils import compute_path_priority
        
        doc_score = compute_path_priority("docs/README.md")
        src_score = compute_path_priority("src/main.py")
        
        assert doc_score < src_score
    
    def test_examples_directory_low_priority(self):
        """Files in examples/ should have low priority."""
        from context_packer.git_utils import compute_path_priority
        
        example_score = compute_path_priority("examples/demo.py")
        src_score = compute_path_priority("src/main.py")
        
        assert example_score < src_score
    
    def test_github_directory_lowest_priority(self):
        """Files in .github/ should have very low priority."""
        from context_packer.git_utils import compute_path_priority
        
        github_score = compute_path_priority(".github/workflows/ci.yml")
        test_score = compute_path_priority("tests/test_main.py")
        
        # .github/ gets PRIORITY_INFRA (-10), lower than tests
        assert github_score < test_score
    
    def test_source_extensions_boost(self):
        """Source code extensions should get a boost over non-source."""
        from context_packer.git_utils import compute_path_priority
        
        py_score = compute_path_priority("main.py")
        md_score = compute_path_priority("README.md")
        
        assert py_score > md_score
    
    def test_cmd_pkg_go_conventions(self):
        """Go convention directories (cmd/, pkg/) should be high priority."""
        from context_packer.git_utils import compute_path_priority
        
        cmd_score = compute_path_priority("cmd/server/main.go")
        pkg_score = compute_path_priority("pkg/utils/helper.go")
        
        assert cmd_score >= 10
        assert pkg_score >= 10
    
    def test_app_directory_high_priority(self):
        """Files in app/ (Rails, some Python) should have high priority."""
        from context_packer.git_utils import compute_path_priority
        
        score = compute_path_priority("app/models/user.py")
        
        assert score >= 10


class TestPrioritySortPaths:
    """Tests for sorting file index by priority."""
    
    def test_sorts_by_priority_descending(self, sample_file_index):
        """Higher priority files should come first."""
        from context_packer.git_utils import priority_sort_paths
        
        sorted_index = priority_sort_paths(sample_file_index)
        
        # Get paths in sorted order
        paths = [entry["path"] for entry in sorted_index]
        
        # src/ files should appear before test/ files
        src_positions = [i for i, p in enumerate(paths) if p.startswith("src/")]
        test_positions = [i for i, p in enumerate(paths) if p.startswith("tests/")]
        
        if src_positions and test_positions:
            assert min(src_positions) < min(test_positions)
    
    def test_source_before_docs(self, sample_file_index):
        """Source code should appear before documentation."""
        from context_packer.git_utils import priority_sort_paths
        
        sorted_index = priority_sort_paths(sample_file_index)
        paths = [entry["path"] for entry in sorted_index]
        
        # Find first .py file and README.md positions
        py_positions = [i for i, p in enumerate(paths) if p.endswith(".py") and "test" not in p.lower()]
        doc_positions = [i for i, p in enumerate(paths) if "docs/" in p or p == "README.md"]
        
        if py_positions and doc_positions:
            assert min(py_positions) < min(doc_positions)
    
    def test_github_last(self, sample_file_index):
        """Infrastructure files should appear last."""
        from context_packer.git_utils import priority_sort_paths
        
        sorted_index = priority_sort_paths(sample_file_index)
        paths = [entry["path"] for entry in sorted_index]
        
        github_positions = [i for i, p in enumerate(paths) if ".github" in p]
        
        # .github should be in bottom half
        if github_positions:
            assert min(github_positions) > len(paths) // 2
    
    def test_preserves_all_entries(self, sample_file_index):
        """Sorting should not lose any entries."""
        from context_packer.git_utils import priority_sort_paths
        
        sorted_index = priority_sort_paths(sample_file_index)
        
        assert len(sorted_index) == len(sample_file_index)
        
        original_paths = {e["path"] for e in sample_file_index}
        sorted_paths = {e["path"] for e in sorted_index}
        
        assert original_paths == sorted_paths


class TestBuildFileIndex:
    """Tests for building file index from a directory."""
    
    def test_indexes_source_files(self, temp_repo):
        """Should index Python and JS source files."""
        from context_packer.git_utils import build_file_index
        
        index = build_file_index(temp_repo)
        paths = {entry["path"] for entry in index}
        
        assert "main.py" in paths
        assert "src/app.py" in paths
        assert "lib/utils.js" in paths
    
    def test_excludes_git_directory(self, temp_repo):
        """Should not index .git/ contents."""
        from context_packer.git_utils import build_file_index
        
        # Create .git directory
        git_dir = temp_repo / ".git"
        git_dir.mkdir()
        (git_dir / "config").write_text("git config")
        
        index = build_file_index(temp_repo)
        paths = {entry["path"] for entry in index}
        
        assert not any(".git" in p for p in paths)
    
    def test_excludes_node_modules(self, temp_repo):
        """Should not index node_modules/."""
        from context_packer.git_utils import build_file_index
        
        # Create node_modules
        nm = temp_repo / "node_modules" / "some-package"
        nm.mkdir(parents=True)
        (nm / "index.js").write_text("module.exports = {}")
        
        index = build_file_index(temp_repo)
        paths = {entry["path"] for entry in index}
        
        assert not any("node_modules" in p for p in paths)
    
    def test_excludes_pycache(self, temp_repo):
        """Should not index __pycache__/."""
        from context_packer.git_utils import build_file_index
        
        # Create __pycache__
        cache = temp_repo / "__pycache__"
        cache.mkdir()
        (cache / "module.cpython-311.pyc").write_bytes(b"compiled")
        
        index = build_file_index(temp_repo)
        paths = {entry["path"] for entry in index}
        
        assert not any("__pycache__" in p for p in paths)
    
    def test_excludes_binary_files(self, temp_repo):
        """Should not index binary file extensions."""
        from context_packer.git_utils import build_file_index
        
        # Create binary files
        (temp_repo / "image.png").write_bytes(b"\x89PNG")
        (temp_repo / "data.db").write_bytes(b"sqlite")
        
        index = build_file_index(temp_repo)
        paths = {entry["path"] for entry in index}
        
        assert "image.png" not in paths
        assert "data.db" not in paths
    
    def test_includes_file_metadata(self, temp_repo):
        """Each entry should have path, size, depth, extension."""
        from context_packer.git_utils import build_file_index
        
        index = build_file_index(temp_repo)
        
        assert len(index) > 0
        
        entry = index[0]
        assert "path" in entry
        assert "size_bytes" in entry or "size" in entry
        # Check for depth or extension if present
    
    def test_respects_max_depth(self, temp_repo):
        """Should not index beyond configured max depth (from settings)."""
        from context_packer.git_utils import build_file_index
        from context_packer.config import settings
        
        # Create deeply nested file beyond MAX_TREE_DEPTH
        max_depth = settings.MAX_TREE_DEPTH
        deep = temp_repo
        for i in range(max_depth + 3):
            deep = deep / f"level_{i}"
        deep.mkdir(parents=True)
        (deep / "deep.py").write_text("# too deep")
        
        index = build_file_index(temp_repo)
        paths = {entry["path"] for entry in index}
        
        # File beyond MAX_TREE_DEPTH should be excluded
        assert not any(p.count("/") > max_depth for p in paths)


class TestFilterValidPaths:
    """Tests for filtering selected paths against file index."""
    
    def test_filters_invalid_paths(self):
        """Should remove paths not in the file index."""
        from context_packer.git_utils import filter_valid_paths
        
        selected = ["src/real.py", "src/fake.py", "lib/core.js"]
        file_index = [
            {"path": "src/real.py"},
            {"path": "lib/core.js"},
            {"path": "tests/test.py"},
        ]
        
        result = filter_valid_paths(selected, file_index)
        
        assert "src/real.py" in result
        assert "lib/core.js" in result
        assert "src/fake.py" not in result
    
    def test_preserves_order(self):
        """Should maintain original selection order."""
        from context_packer.git_utils import filter_valid_paths
        
        selected = ["c.py", "a.py", "b.py"]
        file_index = [
            {"path": "a.py"},
            {"path": "b.py"},
            {"path": "c.py"},
        ]
        
        result = filter_valid_paths(selected, file_index)
        
        assert result == ["c.py", "a.py", "b.py"]
    
    def test_handles_empty_selection(self):
        """Should handle empty selection gracefully."""
        from context_packer.git_utils import filter_valid_paths
        
        result = filter_valid_paths([], [{"path": "file.py"}])
        
        assert result == []
    
    def test_handles_empty_index(self):
        """Should return empty for empty file index."""
        from context_packer.git_utils import filter_valid_paths
        
        result = filter_valid_paths(["file.py"], [])
        
        assert result == []


class TestValidateRepoSize:
    """Tests for repository size validation."""
    
    def test_accepts_normal_repo(self):
        """Should accept repos within size limits."""
        from context_packer.git_utils import validate_repo_size
        
        file_index = [{"path": f"file_{i}.py", "size_bytes": 1000} for i in range(100)]
        
        # Should not raise
        validate_repo_size(file_index)
    
    def test_rejects_too_many_files(self):
        """Should reject repos with too many files."""
        from context_packer.git_utils import validate_repo_size, MAX_FILES_LIMIT
        
        file_index = [{"path": f"file_{i}.py", "size_bytes": 100} for i in range(MAX_FILES_LIMIT + 1)]
        
        with pytest.raises(RuntimeError):
            validate_repo_size(file_index)
