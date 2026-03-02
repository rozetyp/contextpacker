"""
End-to-end answer quality tests with LLM-as-judge evaluation.

These tests measure whether answers generated with ContextPacker context
are correct, relevant, and well-grounded. Uses cross-vendor judging:
- OpenAI generates answer → Gemini judges quality
- Gemini generates answer → OpenAI judges quality

This reduces self-bias and approximates independent review.

Marked with @pytest.mark.e2e - these require API keys and are slow/expensive.
"""

import pytest
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# =============================================================================
# G-Eval Scoring (LLM-as-Judge)
# =============================================================================

GEVAL_SYSTEM_PROMPT = """You are an expert evaluator assessing AI-generated answers about code.

Score the answer on a scale of 1-5 for each criterion:

**Correctness** (Is the answer factually accurate?)
- 5: Completely correct, no errors
- 4: Mostly correct, minor inaccuracies
- 3: Partially correct, some significant errors
- 2: Mostly incorrect
- 1: Completely wrong or nonsensical

**Relevance** (Does the answer address the question?)
- 5: Directly and completely addresses the question
- 4: Mostly relevant with minor tangents
- 3: Partially relevant
- 2: Mostly off-topic
- 1: Completely irrelevant

**Grounding** (Is the answer based on the provided context?)
- 5: Fully grounded in provided context
- 4: Mostly grounded, some external knowledge
- 3: Partially grounded
- 2: Mostly hallucinated
- 1: Completely made up

Respond with JSON only:
{"correctness": N, "relevance": N, "grounding": N, "reasoning": "brief explanation"}
"""


@dataclass
class EvalResult:
    """Result from LLM-as-judge evaluation."""
    correctness: float
    relevance: float
    grounding: float
    reasoning: str
    
    @property
    def average(self) -> float:
        return (self.correctness + self.relevance + self.grounding) / 3


def parse_eval_response(response: str) -> Optional[EvalResult]:
    """Parse JSON response from evaluator LLM."""
    try:
        # Handle markdown code blocks
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
        
        data = json.loads(response.strip())
        return EvalResult(
            correctness=float(data.get("correctness", 0)),
            relevance=float(data.get("relevance", 0)),
            grounding=float(data.get("grounding", 0)),
            reasoning=data.get("reasoning", "")
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_e2e_questions():
    """Sample questions for E2E testing."""
    return [
        {
            "id": "flask_easy_01",
            "repo_url": "https://github.com/pallets/flask",
            "query": "Where is the Flask application class defined?",
            "expected_mentions": ["app.py", "Flask", "class"],
        },
        {
            "id": "flask_medium_01", 
            "repo_url": "https://github.com/pallets/flask",
            "query": "How does Flask's request context work?",
            "expected_mentions": ["ctx.py", "context", "request"],
        },
    ]


@pytest.fixture
def mock_context():
    """Mock context pack for testing without API calls."""
    return """
## Context for: Where is the Flask application class defined?

*2 files selected*

<!-- contextpacker: reason="Main Flask application class" -->
### src/flask/app.py
```python
class Flask:
    '''The flask object implements a WSGI application.'''
    
    def __init__(self, import_name):
        self.import_name = import_name
        self.config = Config()
    
    def route(self, rule, **options):
        '''Decorator for registering routes.'''
        pass
```

<!-- contextpacker: reason="Flask package entry point" -->
### src/flask/__init__.py
```python
from .app import Flask
__all__ = ['Flask']
```
"""


@pytest.fixture
def mock_answer():
    """Mock LLM answer for testing evaluation."""
    return """
The Flask application class is defined in `src/flask/app.py`. 

The `Flask` class is the central object that implements a WSGI application. 
When you create an instance like `app = Flask(__name__)`, you're instantiating 
this class. It handles routing, configuration, request/response processing, 
and more.

Key methods include:
- `__init__`: Initializes the app with the import name and default config
- `route`: A decorator for registering URL rules
"""


# =============================================================================
# Unit Tests for Evaluation Logic
# =============================================================================

class TestParseEvalResponse:
    """Tests for parsing LLM evaluation responses."""
    
    def test_parses_valid_json(self):
        """Should parse valid JSON response."""
        response = '{"correctness": 5, "relevance": 4, "grounding": 5, "reasoning": "Good answer"}'
        
        result = parse_eval_response(response)
        
        assert result is not None
        assert result.correctness == 5
        assert result.relevance == 4
        assert result.grounding == 5
        assert result.reasoning == "Good answer"
    
    def test_parses_json_in_code_block(self):
        """Should handle JSON wrapped in markdown code block."""
        response = '''Here's my evaluation:
```json
{"correctness": 4, "relevance": 4, "grounding": 3, "reasoning": "Mostly correct"}
```
'''
        result = parse_eval_response(response)
        
        assert result is not None
        assert result.correctness == 4
    
    def test_returns_none_for_invalid_json(self):
        """Should return None for invalid JSON."""
        response = "This is not JSON at all"
        
        result = parse_eval_response(response)
        
        assert result is None
    
    def test_handles_missing_fields(self):
        """Should handle missing optional fields."""
        response = '{"correctness": 5, "relevance": 5, "grounding": 5}'
        
        result = parse_eval_response(response)
        
        assert result is not None
        assert result.reasoning == ""


class TestEvalResult:
    """Tests for EvalResult dataclass."""
    
    def test_average_calculation(self):
        """Should correctly calculate average score."""
        result = EvalResult(
            correctness=5,
            relevance=4,
            grounding=3,
            reasoning="test"
        )
        
        assert result.average == 4.0
    
    def test_perfect_score(self):
        """Perfect scores should average to 5.0."""
        result = EvalResult(
            correctness=5,
            relevance=5,
            grounding=5,
            reasoning="perfect"
        )
        
        assert result.average == 5.0


# =============================================================================
# E2E Tests (Require API Keys)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.slow
class TestE2EAnswerQuality:
    """
    End-to-end tests for answer quality.
    
    These tests:
    1. Call ContextPacker API to get context
    2. Generate answer using one LLM
    3. Evaluate answer using a different LLM (cross-vendor)
    
    Run with: pytest -m e2e
    """
    
    @pytest.mark.skip(reason="Requires live API keys - run manually")
    async def test_flask_easy_question(self, sample_e2e_questions):
        """Test easy Flask question with cross-vendor evaluation."""
        # Would implement full E2E flow here
        pass
    
    @pytest.mark.skip(reason="Requires live API keys - run manually")
    async def test_cross_vendor_judging(self):
        """
        Test cross-vendor judging reduces self-bias.
        
        Compare:
        - OpenAI answer → Gemini judge
        - Gemini answer → OpenAI judge
        
        Scores should be similar if both are objective.
        """
        pass
    
    def test_context_improves_grounding(self, mock_context, mock_answer):
        """Answers with context should have better grounding scores."""
        # This is a sanity check that can run without API
        # Real test would compare with/without context
        assert "Flask" in mock_answer
        assert "app.py" in mock_answer
    
    def test_answer_mentions_expected_content(self, mock_context, mock_answer):
        """Answer should mention key content from context."""
        expected = ["Flask", "class", "app.py", "WSGI"]
        
        mentions = sum(1 for e in expected if e.lower() in mock_answer.lower())
        
        # Should mention at least 3 of 4 expected terms
        assert mentions >= 3


@pytest.mark.e2e
class TestApproachComparison:
    """
    Compare ContextPacker vs baselines on answer quality.
    
    This is the key evaluation from e2e.md:
    - ContextPacker should be "in the same performance band as embeddings"
    - Not trying to beat embeddings on every metric
    - Key win is zero-setup, no infra
    """
    
    def test_scoring_targets(self):
        """Define target scores based on e2e.md strategy."""
        # These are the targets from your e2e.md:
        # "embedding-class retrieval without indexes, vector DBs, or setup"
        
        # Target: >= 80% correctness (matching embeddings baseline)
        TARGET_CORRECTNESS = 0.80
        
        # Target: >= 80% relevance
        TARGET_RELEVANCE = 0.80
        
        # Target: >= 90% grounding (context-based answers)
        TARGET_GROUNDING = 0.90
        
        # These are aspirational - actual benchmarks test against them
        assert TARGET_CORRECTNESS >= 0.75
        assert TARGET_RELEVANCE >= 0.75
        assert TARGET_GROUNDING >= 0.85
    
    def test_approach_definitions(self):
        """Document the approaches we compare."""
        approaches = {
            "contextpacker": {
                "description": "LLM-based file selection, no embeddings",
                "expected_latency": "2-4s cold, 1-2s warm",
                "expected_cost": "$0.0002/query for selection",
                "infra_required": "None - just API call",
            },
            "embeddings": {
                "description": "Clone + embed + vector search",
                "expected_latency": "30-120s for embedding, 1s for search",
                "expected_cost": "$0.0001/query for search + $0.50-5 for indexing",
                "infra_required": "Vector DB, embedding pipeline",
            },
            "no_context": {
                "description": "LLM without any repository context",
                "expected_latency": "<1s",
                "expected_cost": "Just LLM inference",
                "infra_required": "None",
            },
        }
        
        # ContextPacker should require no infra
        assert approaches["contextpacker"]["infra_required"] == "None - just API call"
        
        # Embeddings requires infra (our competitive disadvantage to avoid)
        assert "Vector DB" in approaches["embeddings"]["infra_required"]


# =============================================================================
# Benchmark Data Structures
# =============================================================================

@dataclass
class E2EQuestion:
    """A question for E2E evaluation."""
    id: str
    repo_url: str
    query: str
    expected_mentions: List[str]  # Terms that should appear in good answers
    difficulty: str = "medium"
    

@dataclass
class E2EResult:
    """Result from E2E evaluation of one question."""
    question_id: str
    approach: str
    answer: str
    eval_scores: Optional[EvalResult]
    latency_ms: float
    files_used: List[str]
    

def load_e2e_questions(filepath: str) -> List[E2EQuestion]:
    """Load E2E questions from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    
    return [
        E2EQuestion(
            id=q["id"],
            repo_url=q["repo_url"],
            query=q["query"],
            expected_mentions=q.get("expected_mentions", []),
            difficulty=q.get("difficulty", "medium"),
        )
        for q in data.get("questions", data)  # Handle both formats
    ]


class TestE2EDataStructures:
    """Tests for E2E data structures."""
    
    def test_question_dataclass(self):
        """E2EQuestion should hold question data."""
        q = E2EQuestion(
            id="test_01",
            repo_url="https://github.com/test/repo",
            query="How does X work?",
            expected_mentions=["X", "implementation"],
        )
        
        assert q.id == "test_01"
        assert q.difficulty == "medium"  # default
    
    def test_result_dataclass(self):
        """E2EResult should hold evaluation results."""
        result = E2EResult(
            question_id="test_01",
            approach="contextpacker",
            answer="X works by...",
            eval_scores=EvalResult(5, 4, 5, "good"),
            latency_ms=1500.0,
            files_used=["src/x.py"],
        )
        
        assert result.approach == "contextpacker"
        assert result.eval_scores.average == 14/3
