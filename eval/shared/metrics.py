"""
Industry-standard retrieval metrics.

Based on CodeSearchNet benchmark (GitHub, 2019).
Extended with RAG-specific metrics per EVALUATION_STRATEGY.md.
"""

import math
import random
from typing import List, Set, Dict, Tuple, Callable


def dcg_at_k(relevances: List[float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def ndcg_at_k(selected: List[str], ground_truth: Set[str], alternatives: Set[str], k: int = 10) -> float:
    """
    Normalized Discounted Cumulative Gain at K.
    
    Primary metric used in CodeSearchNet benchmark.
    Assigns grade 3 to ground_truth, grade 1 to alternatives.
    """
    if not ground_truth:
        return 1.0
    
    # Build relevance grades: ground_truth=3, alternatives=1
    grades = {f: 3.0 for f in ground_truth}
    grades.update({f: 1.0 for f in alternatives if f not in grades})
    
    # Get relevances for selected items
    relevances = [grades.get(f, 0.0) for f in selected[:k]]
    
    # DCG
    dcg = dcg_at_k(relevances, k)
    
    # Ideal DCG
    ideal = sorted(grades.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal, k)
    
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(selected: List[str], ground_truth: Set[str], alternatives: Set[str], k: int = 10) -> float:
    """Did we find at least one relevant file in top K?"""
    if not ground_truth:
        return 1.0
    acceptable = ground_truth | alternatives
    top_k = set(selected[:k])
    return 1.0 if top_k & acceptable else 0.0


def mrr(selected: List[str], ground_truth: Set[str], alternatives: Set[str]) -> float:
    """Mean Reciprocal Rank - how early is the first relevant file?"""
    if not ground_truth:
        return 1.0
    acceptable = ground_truth | alternatives
    for i, f in enumerate(selected):
        if f in acceptable:
            return 1.0 / (i + 1)
    return 0.0


# =============================================================================
# NEW METRICS (per EVALUATION_STRATEGY.md)
# =============================================================================

def recall_at_k(selected: List[str], ground_truth: Set[str], alternatives: Set[str], k: int = 10) -> float:
    """
    What fraction of relevant files did we retrieve in top-k?
    
    Uses ground_truth only (not alternatives) for recall calculation.
    """
    if not ground_truth:
        return 1.0
    
    top_k = set(selected[:k])
    found = top_k & ground_truth
    return len(found) / len(ground_truth)


def precision_at_k(selected: List[str], ground_truth: Set[str], alternatives: Set[str], k: int = 10) -> float:
    """
    What fraction of top-k files are relevant?
    
    Counts both ground_truth and alternatives as relevant.
    """
    if not selected[:k]:
        return 0.0
    
    top_k = selected[:k]
    acceptable = ground_truth | alternatives
    relevant_count = sum(1 for f in top_k if f in acceptable)
    return relevant_count / len(top_k)


def all_critical_at_k(selected: List[str], ground_truth: Set[str], k: int = 10) -> float:
    """
    Did we retrieve ALL critical (ground_truth) files in top-k?
    
    For multi-file answers, partial retrieval isn't enough.
    Returns 1.0 if all ground_truth files are in top-k, else 0.0.
    """
    if not ground_truth:
        return 1.0
    
    top_k = set(selected[:k])
    return 1.0 if ground_truth <= top_k else 0.0


def junk_ratio_at_k(selected: List[str], ground_truth: Set[str], alternatives: Set[str], k: int = 10) -> float:
    """
    What fraction of top-k files are irrelevant (junk)?
    
    Catches "spray 30 files, hope one is right" behavior.
    Lower is better. 0.0 = all files relevant. 1.0 = all files junk.
    """
    if not selected[:k]:
        return 0.0
    
    top_k = selected[:k]
    acceptable = ground_truth | alternatives
    junk_count = sum(1 for f in top_k if f not in acceptable)
    return junk_count / len(top_k)


# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

def bootstrap_ci(
    values: List[float], 
    n_bootstrap: int = 1000, 
    ci: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a list of metric values.
    
    Args:
        values: Per-query metric values
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval (0.95 = 95%)
    
    Returns:
        (mean, lower_bound, upper_bound)
    """
    if not values:
        return (0.0, 0.0, 0.0)
    
    n = len(values)
    means = []
    
    for _ in range(n_bootstrap):
        sample = [random.choice(values) for _ in range(n)]
        means.append(sum(sample) / len(sample))
    
    means.sort()
    alpha = (1 - ci) / 2
    lower_idx = int(alpha * n_bootstrap)
    upper_idx = int((1 - alpha) * n_bootstrap) - 1
    
    mean = sum(values) / len(values)
    return (mean, means[lower_idx], means[upper_idx])


def format_ci(mean: float, lower: float, upper: float, decimals: int = 2) -> str:
    """Format a confidence interval as 'mean [lower, upper]'."""
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(mean)} [{fmt.format(lower)}, {fmt.format(upper)}]"


# =============================================================================
# METRIC AGGREGATION
# =============================================================================

def compute_all_metrics(
    selected: List[str], 
    ground_truth: Set[str], 
    alternatives: Set[str],
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Compute all metrics for a single query.
    
    Returns dict with keys like 'hit@5', 'hit@10', 'mrr', 'ndcg@10', etc.
    """
    results = {}
    
    # MRR (doesn't depend on k)
    results['mrr'] = mrr(selected, ground_truth, alternatives)
    
    for k in k_values:
        results[f'hit@{k}'] = hit_at_k(selected, ground_truth, alternatives, k)
        results[f'ndcg@{k}'] = ndcg_at_k(selected, ground_truth, alternatives, k)
        results[f'recall@{k}'] = recall_at_k(selected, ground_truth, alternatives, k)
        results[f'precision@{k}'] = precision_at_k(selected, ground_truth, alternatives, k)
        results[f'all_critical@{k}'] = all_critical_at_k(selected, ground_truth, k)
        results[f'junk_ratio@{k}'] = junk_ratio_at_k(selected, ground_truth, alternatives, k)
    
    return results


# Industry baselines for reference
INDUSTRY_BASELINES = {
    "bm25": {"ndcg": 0.31, "mrr": 0.40},
    "codebert": {"ndcg": 0.69, "mrr": 0.72},
    "unixcoder": {"ndcg": 0.75, "mrr": 0.78},
}
