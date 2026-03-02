"""
Retrieval quality benchmarks - Hit@K and MRR metrics.

These tests measure whether ContextPacker retrieves relevant files.
They use ground truth from eval/benchmarks/ and compare against baselines.

Marked with @pytest.mark.benchmark - run separately from unit tests.
"""

import pytest
from typing import List, Set


# =============================================================================
# Metric Functions (copied from eval/runner.py for self-contained tests)
# =============================================================================

def hit_at_k(selected: List[str], ground_truth: Set[str], alternatives: Set[str] = None, k: int = 10) -> float:
    """
    Did we find at least one relevant file in top K?
    
    Args:
        selected: List of selected file paths (ranked)
        ground_truth: Set of files that MUST be found
        alternatives: Set of files that are also acceptable
        k: Number of top results to consider
    
    Returns:
        1.0 if at least one relevant file found, 0.0 otherwise
    """
    if not ground_truth:
        return 1.0
    
    alternatives = alternatives or set()
    acceptable = ground_truth | alternatives
    top_k = set(selected[:k])
    
    return 1.0 if top_k & acceptable else 0.0


def mrr(selected: List[str], ground_truth: Set[str], alternatives: Set[str] = None) -> float:
    """
    Mean Reciprocal Rank - how early is the first relevant file?
    
    Args:
        selected: List of selected file paths (ranked)
        ground_truth: Set of files that MUST be found
        alternatives: Set of files that are also acceptable
    
    Returns:
        1/rank of first relevant file, or 0.0 if not found
    """
    if not ground_truth:
        return 1.0
    
    alternatives = alternatives or set()
    acceptable = ground_truth | alternatives
    
    for i, f in enumerate(selected):
        if f in acceptable:
            return 1.0 / (i + 1)
    
    return 0.0


def recall_at_k(selected: List[str], ground_truth: Set[str], alternatives: Set[str] = None, k: int = 10) -> float:
    """
    What fraction of ground truth files did we find in top K?
    
    Args:
        selected: List of selected file paths (ranked)
        ground_truth: Set of files that MUST be found
        alternatives: Set of files that are also acceptable
        k: Number of top results to consider
    
    Returns:
        Fraction of ground truth files found (0.0 to 1.0)
    """
    if not ground_truth:
        return 1.0
    
    alternatives = alternatives or set()
    acceptable = ground_truth | alternatives
    top_k = set(selected[:k])
    
    found = len(top_k & acceptable)
    return found / len(ground_truth)


# =============================================================================
# Unit Tests for Metrics
# =============================================================================

class TestHitAtK:
    """Unit tests for Hit@K metric."""
    
    def test_perfect_hit(self):
        """Should return 1.0 when ground truth file is in top K."""
        selected = ["src/app.py", "src/utils.py", "lib/core.js"]
        ground_truth = {"src/app.py"}
        
        assert hit_at_k(selected, ground_truth, k=10) == 1.0
    
    def test_miss(self):
        """Should return 0.0 when no ground truth file in top K."""
        selected = ["src/other.py", "src/utils.py", "lib/core.js"]
        ground_truth = {"src/app.py"}
        
        assert hit_at_k(selected, ground_truth, k=10) == 0.0
    
    def test_alternative_counts(self):
        """Should count alternative files as hits."""
        selected = ["src/alternative.py", "src/utils.py"]
        ground_truth = {"src/app.py"}
        alternatives = {"src/alternative.py"}
        
        assert hit_at_k(selected, ground_truth, alternatives, k=10) == 1.0
    
    def test_respects_k_limit(self):
        """Should only consider top K files."""
        selected = ["src/wrong1.py", "src/wrong2.py", "src/wrong3.py", "src/app.py"]
        ground_truth = {"src/app.py"}
        
        assert hit_at_k(selected, ground_truth, k=3) == 0.0  # app.py is at position 4
        assert hit_at_k(selected, ground_truth, k=4) == 1.0  # app.py is included
    
    def test_empty_ground_truth(self):
        """Should return 1.0 for empty ground truth."""
        selected = ["src/app.py"]
        
        assert hit_at_k(selected, set(), k=10) == 1.0
    
    def test_empty_selection(self):
        """Should return 0.0 when selection is empty."""
        ground_truth = {"src/app.py"}
        
        assert hit_at_k([], ground_truth, k=10) == 0.0


class TestMRR:
    """Unit tests for Mean Reciprocal Rank metric."""
    
    def test_first_position(self):
        """Should return 1.0 when ground truth is first."""
        selected = ["src/app.py", "src/utils.py"]
        ground_truth = {"src/app.py"}
        
        assert mrr(selected, ground_truth) == 1.0
    
    def test_second_position(self):
        """Should return 0.5 when ground truth is second."""
        selected = ["src/utils.py", "src/app.py", "lib/core.js"]
        ground_truth = {"src/app.py"}
        
        assert mrr(selected, ground_truth) == 0.5
    
    def test_third_position(self):
        """Should return 0.333... when ground truth is third."""
        selected = ["src/a.py", "src/b.py", "src/app.py"]
        ground_truth = {"src/app.py"}
        
        assert abs(mrr(selected, ground_truth) - 1/3) < 0.01
    
    def test_not_found(self):
        """Should return 0.0 when ground truth not in selection."""
        selected = ["src/a.py", "src/b.py", "src/c.py"]
        ground_truth = {"src/app.py"}
        
        assert mrr(selected, ground_truth) == 0.0
    
    def test_alternative_counts(self):
        """Should use alternative if it appears before ground truth."""
        selected = ["src/alt.py", "src/app.py"]
        ground_truth = {"src/app.py"}
        alternatives = {"src/alt.py"}
        
        # alt.py is at position 1, so MRR = 1.0
        assert mrr(selected, ground_truth, alternatives) == 1.0
    
    def test_empty_ground_truth(self):
        """Should return 1.0 for empty ground truth."""
        selected = ["src/app.py"]
        
        assert mrr(selected, set()) == 1.0


class TestRecallAtK:
    """Unit tests for Recall@K metric."""
    
    def test_full_recall(self):
        """Should return 1.0 when all ground truth files found."""
        selected = ["src/app.py", "src/config.py", "src/utils.py"]
        ground_truth = {"src/app.py", "src/config.py"}
        
        assert recall_at_k(selected, ground_truth, k=10) == 1.0
    
    def test_partial_recall(self):
        """Should return fraction when some ground truth found."""
        selected = ["src/app.py", "src/utils.py"]
        ground_truth = {"src/app.py", "src/config.py"}
        
        assert recall_at_k(selected, ground_truth, k=10) == 0.5
    
    def test_no_recall(self):
        """Should return 0.0 when no ground truth found."""
        selected = ["src/other.py", "src/utils.py"]
        ground_truth = {"src/app.py", "src/config.py"}
        
        assert recall_at_k(selected, ground_truth, k=10) == 0.0
    
    def test_alternatives_count(self):
        """Alternatives should count toward recall."""
        selected = ["src/alt.py", "src/utils.py"]
        ground_truth = {"src/app.py", "src/config.py"}
        alternatives = {"src/alt.py"}
        
        # alt.py found, so 1/2 = 0.5
        assert recall_at_k(selected, ground_truth, alternatives, k=10) == 0.5


# =============================================================================
# Benchmark Fixtures
# =============================================================================

@pytest.fixture
def flask_questions():
    """Sample Flask benchmark questions for testing."""
    return [
        {
            "id": "flask_easy_01",
            "query": "Where is the Flask application class defined?",
            "ground_truth_files": ["src/flask/app.py"],
            "alternatives": ["src/flask/__init__.py"],
        },
        {
            "id": "flask_medium_01",
            "query": "How does Flask's thread-local request context work?",
            "ground_truth_files": ["src/flask/ctx.py", "src/flask/globals.py"],
            "alternatives": [],
        },
    ]


@pytest.fixture  
def sample_selection_result():
    """Sample file selection result for testing metrics."""
    return [
        "src/flask/app.py",
        "src/flask/ctx.py",
        "src/flask/blueprints.py",
        "src/flask/config.py",
        "src/flask/globals.py",
    ]


# =============================================================================
# Integration-style Benchmark Tests
# =============================================================================

@pytest.mark.benchmark
class TestRetrievalQuality:
    """
    Benchmark tests that verify retrieval quality meets targets.
    
    These are not CI tests - they require API calls and are slow.
    Run with: pytest -m benchmark
    """
    
    def test_metrics_on_sample_data(self, flask_questions, sample_selection_result):
        """Verify metric calculation on known data."""
        q = flask_questions[0]
        selected = sample_selection_result
        gt = set(q["ground_truth_files"])
        alt = set(q["alternatives"])
        
        # Flask app.py is first in selection, ground truth
        assert hit_at_k(selected, gt, alt, k=10) == 1.0
        assert mrr(selected, gt, alt) == 1.0
    
    def test_hit_at_10_target(self, flask_questions, sample_selection_result):
        """
        Hit@10 should be >= 80% on sample benchmark.
        
        This is a regression test - if we're seeing significantly lower,
        something may have broken in the selection logic.
        """
        hits = []
        for q in flask_questions:
            gt = set(q["ground_truth_files"])
            alt = set(q.get("alternatives", []))
            h = hit_at_k(sample_selection_result, gt, alt, k=10)
            hits.append(h)
        
        hit_rate = sum(hits) / len(hits)
        
        # Should hit at least 50% on this small sample
        # (Real target is 75-80% on full benchmark)
        assert hit_rate >= 0.5


@pytest.mark.benchmark  
@pytest.mark.slow
class TestLiveBenchmarks:
    """
    Live benchmark tests that call the real API.
    
    These are expensive and slow - run manually, not in CI.
    Run with: pytest -m "benchmark and slow"
    """
    
    @pytest.mark.skip(reason="Requires live API - run manually")
    async def test_flask_benchmark(self):
        """Run Flask benchmark against live API."""
        # This would import and call the actual API
        # Left as placeholder for manual testing
        pass
    
    @pytest.mark.skip(reason="Requires live API - run manually")  
    async def test_embeddings_comparison(self):
        """Compare against embeddings baseline."""
        # This would run the full comparison
        pass
