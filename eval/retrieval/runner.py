#!/usr/bin/env python3
"""
Retrieval Benchmark: File Selection Quality

Measures: Does ContextPacker select the right files?
Metrics: NDCG@10, MRR, Hit@10 (CodeSearchNet standard)
Baselines: BM25 (lexical), Embeddings (pre-indexed dense retrieval)

Usage:
    python -m eval.retrieval.runner --help
    python -m eval.retrieval.runner --repo flask --no-embeddings
    python -m eval.retrieval.runner --all --no-embeddings
    
    # Pre-index repos first for fair embeddings comparison:
    python -m eval.shared.embeddings --all
    python -m eval.retrieval.runner --repo flask
"""

import argparse
import asyncio
import json
import os
import re
import math
import time
import httpx
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Set, Tuple, Optional, Dict, Any
from collections import Counter

# Add parent to path for shared imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from eval.shared.metrics import (
    ndcg_at_k, mrr, hit_at_k,
    recall_at_k, precision_at_k, all_critical_at_k, junk_ratio_at_k,
    bootstrap_ci, format_ci
)
from eval.shared.clone import clone_repo, get_code_files
from eval.shared import embeddings as emb_search  # Pre-indexed embeddings

# k values to evaluate (per EVALUATION_STRATEGY.md)
K_VALUES = [5, 10]

# =============================================================================
# Configuration
# =============================================================================

QUESTIONS_DIR = Path(__file__).parent / "questions"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

API_URL = "https://contextpacker.com/v1/packs"
API_KEY = os.getenv("CONTEXTPACKER_API_KEY", "cp_live_your_key_here")
GITHUB_PAT = os.getenv("GITHUB_PAT", "")


# =============================================================================
# ContextPacker API
# =============================================================================

async def call_contextpacker(repo_url: str, query: str, is_private: bool = False) -> Tuple[List[str], float]:
    """Call ContextPacker API. Returns (files, latency_ms)."""
    payload = {
        "repo_url": repo_url,
        "query": query,
        "max_tokens": 30000
    }
    if is_private and GITHUB_PAT:
        payload["vcs"] = {"token": GITHUB_PAT}
    
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                API_URL,
                headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
                json=payload
            )
            latency = (time.perf_counter() - start) * 1000
            
            if resp.status_code != 200:
                print(f"    ⚠️ CP error: {resp.status_code}")
                return [], latency
            
            data = resp.json()
            files = [f.get("path", "") for f in data.get("files", [])]
            return files, latency
        except Exception as e:
            return [], (time.perf_counter() - start) * 1000


# =============================================================================
# BM25 Baseline
# =============================================================================

def tokenize(text: str) -> List[str]:
    """Simple tokenizer for code."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = text.replace('_', ' ')
    tokens = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', text.lower())
    return [t for t in tokens if len(t) > 1]


class BM25:
    """BM25 implementation for code search."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs = []
        self.doc_freqs = Counter()
        self.avg_dl = 0
        self.N = 0
    
    def index(self, documents: List[Tuple[str, str]]):
        """Index documents: [(path, content), ...]"""
        self.docs = []
        self.doc_freqs = Counter()
        
        for path, content in documents:
            tokens = tokenize(path) + tokenize(content)
            self.docs.append((path, tokens))
            for term in set(tokens):
                self.doc_freqs[term] += 1
        
        self.N = len(self.docs)
        self.avg_dl = sum(len(d[1]) for d in self.docs) / self.N if self.N else 1
    
    def score(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """BM25 score for a single document."""
        score = 0.0
        doc_len = len(doc_tokens)
        term_freqs = Counter(doc_tokens)
        
        for term in query_tokens:
            if term not in term_freqs:
                continue
            
            tf = term_freqs[term]
            df = self.doc_freqs.get(term, 0)
            idf = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
            tf_component = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl))
            score += idf * tf_component
        
        return score
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search for query, return [(path, score), ...]"""
        query_tokens = tokenize(query)
        scored = []
        for path, doc_tokens in self.docs:
            s = self.score(query_tokens, doc_tokens)
            if s > 0:
                scored.append((path, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]


def bm25_search(repo_url: str, query: str) -> Tuple[List[str], float]:
    """BM25-based file search."""
    start = time.perf_counter()
    repo_path = clone_repo(repo_url, GITHUB_PAT)
    if not repo_path:
        return [], (time.perf_counter() - start) * 1000
    
    files = get_code_files(repo_path)
    if not files:
        return [], (time.perf_counter() - start) * 1000
    
    bm25 = BM25()
    bm25.index(files)
    results = bm25.search(query, top_k=10)
    
    return [path for path, _ in results], (time.perf_counter() - start) * 1000


# =============================================================================
# Runner
# =============================================================================

@dataclass
class QuestionResult:
    """Per-question results with all metrics."""
    id: str
    query: str
    difficulty: str
    category: str
    ground_truth: List[str]
    alternatives: List[str]
    # CP metrics
    cp_files: List[str]
    cp_metrics: Dict[str, float]  # hit@5, hit@10, ndcg@10, mrr, recall@10, etc.
    cp_latency_ms: float
    # BM25 metrics
    bm25_files: List[str]
    bm25_metrics: Dict[str, float]
    bm25_latency_ms: float
    # Embeddings metrics
    emb_files: List[str]
    emb_metrics: Dict[str, float]
    emb_latency_ms: float
    # Winner at k=10
    winner: str


def compute_metrics(selected: List[str], gt: Set[str], alt: Set[str]) -> Dict[str, float]:
    """Compute all metrics for a retrieval result."""
    metrics = {"mrr": mrr(selected, gt, alt)}
    for k in K_VALUES:
        metrics[f"hit@{k}"] = hit_at_k(selected, gt, alt, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(selected, gt, alt, k)
        metrics[f"recall@{k}"] = recall_at_k(selected, gt, alt, k)
        metrics[f"precision@{k}"] = precision_at_k(selected, gt, alt, k)
        metrics[f"all_critical@{k}"] = all_critical_at_k(selected, gt, k)
        metrics[f"junk_ratio@{k}"] = junk_ratio_at_k(selected, gt, alt, k)
    return metrics


async def run_benchmark(benchmark_path: str, skip_embeddings: bool = False) -> Dict[str, Any]:
    """Run a single benchmark file with comprehensive metrics."""
    with open(benchmark_path) as f:
        benchmark = json.load(f)
    
    repo_url = benchmark["repo_url"]
    questions = benchmark["questions"]
    name = Path(benchmark_path).stem
    is_private = benchmark.get("is_private", False)
    
    # Determine tier from path
    tier = "unknown"
    for part in Path(benchmark_path).parts:
        if part.startswith("tier"):
            tier = part
            break
    
    print(f"\n{'='*60}")
    print(f"RETRIEVAL BENCHMARK: {name} ({tier})")
    print(f"Repo: {repo_url}")
    print(f"Questions: {len(questions)}")
    print(f"{'='*60}\n")
    
    results = []
    # Collect all metric values for bootstrap CIs
    cp_metrics_all = {m: [] for m in ["hit@5", "hit@10", "ndcg@10", "mrr", "recall@10", "precision@10"]}
    bm25_metrics_all = {m: [] for m in ["hit@5", "hit@10", "ndcg@10", "mrr", "recall@10", "precision@10"]}
    emb_metrics_all = {m: [] for m in ["hit@5", "hit@10", "ndcg@10", "mrr", "recall@10", "precision@10"]}
    cp_lats, bm25_lats, emb_lats = [], [], []
    
    for i, q in enumerate(questions):
        query = q["query"]
        gt = set(q["ground_truth_files"])
        alt = set(q.get("alternatives", []))
        difficulty = q.get("difficulty", "medium")
        category = q.get("category", "unknown")
        
        print(f"[{i+1}/{len(questions)}] [{difficulty.upper()}] {query[:50]}...")
        
        # ContextPacker
        cp_files, cp_lat = await call_contextpacker(repo_url, query, is_private)
        cp_m = compute_metrics(cp_files, gt, alt)
        cp_lats.append(cp_lat)
        for k in cp_metrics_all:
            cp_metrics_all[k].append(cp_m.get(k, 0))
        
        # BM25
        bm25_files, bm25_lat = bm25_search(repo_url, query)
        bm25_m = compute_metrics(bm25_files, gt, alt)
        bm25_lats.append(bm25_lat)
        for k in bm25_metrics_all:
            bm25_metrics_all[k].append(bm25_m.get(k, 0))
        
        # Embeddings (pre-indexed)
        if skip_embeddings:
            emb_files, emb_lat = [], 0
            emb_m = {k: 0 for k in emb_metrics_all}
        else:
            emb_files, emb_lat = emb_search.search(repo_url, query, github_pat=GITHUB_PAT)
            emb_m = compute_metrics(emb_files, gt, alt)
        emb_lats.append(emb_lat)
        for k in emb_metrics_all:
            emb_metrics_all[k].append(emb_m.get(k, 0))
        
        # Winner (based on hit@10 then mrr)
        cp_h, bm25_h = cp_m["hit@10"], bm25_m["hit@10"]
        if cp_h > bm25_h or (cp_h == bm25_h and cp_m["mrr"] > bm25_m["mrr"]):
            winner = "CP"
        elif bm25_h > cp_h or (cp_h == bm25_h and bm25_m["mrr"] > cp_m["mrr"]):
            winner = "BM25"
        else:
            winner = "TIE"
        
        cp_mark = "✓" if cp_h else "✗"
        bm25_mark = "✓" if bm25_h else "✗"
        print(f"    CP: {cp_mark} ndcg={cp_m['ndcg@10']:.2f} | BM25: {bm25_mark} ndcg={bm25_m['ndcg@10']:.2f} | {winner}")
        
        results.append(QuestionResult(
            id=q["id"], query=query, difficulty=difficulty, category=category,
            ground_truth=list(gt), alternatives=list(alt),
            cp_files=cp_files[:10], cp_metrics=cp_m, cp_latency_ms=cp_lat,
            bm25_files=bm25_files[:10], bm25_metrics=bm25_m, bm25_latency_ms=bm25_lat,
            emb_files=emb_files[:10], emb_metrics=emb_m, emb_latency_ms=emb_lat,
            winner=winner
        ))
    
    # Compute summary with bootstrap CIs
    n = len(questions)
    
    def avg_with_ci(values):
        if not values:
            return {"mean": 0, "ci_lower": 0, "ci_upper": 0}
        mean, lower, upper = bootstrap_ci(values, n_bootstrap=1000)
        return {"mean": mean, "ci_lower": lower, "ci_upper": upper}
    
    summary = {
        "benchmark": name,
        "repo": repo_url,
        "tier": tier,
        "questions": n,
        "cp": {k: avg_with_ci(v) for k, v in cp_metrics_all.items()},
        "bm25": {k: avg_with_ci(v) for k, v in bm25_metrics_all.items()},
        "emb": {k: avg_with_ci(v) for k, v in emb_metrics_all.items()} if not skip_embeddings else None,
        "cp_latency_avg_ms": sum(cp_lats) / n if cp_lats else 0,
        "cp_wins": sum(1 for r in results if r.winner == "CP"),
        "bm25_wins": sum(1 for r in results if r.winner == "BM25"),
        "ties": sum(1 for r in results if r.winner == "TIE"),
    }
    
    # Print summary (per EVALUATION_STRATEGY.md format)
    print(f"\n{'─'*70}")
    print(f"RESULTS: {name} ({tier})")
    print(f"{'─'*70}")
    print(f"                Hit@5           Hit@10          MRR             NDCG@10")
    
    cp = summary["cp"]
    print(f"  CP:           {format_ci(*list(cp['hit@5'].values()))}  "
          f"{format_ci(*list(cp['hit@10'].values()))}  "
          f"{format_ci(*list(cp['mrr'].values()))}  "
          f"{format_ci(*list(cp['ndcg@10'].values()))}")
    
    bm = summary["bm25"]
    print(f"  BM25:         {format_ci(*list(bm['hit@5'].values()))}  "
          f"{format_ci(*list(bm['hit@10'].values()))}  "
          f"{format_ci(*list(bm['mrr'].values()))}  "
          f"{format_ci(*list(bm['ndcg@10'].values()))}")
    
    if summary["emb"]:
        em = summary["emb"]
        print(f"  Embeddings:   {format_ci(*list(em['hit@5'].values()))}  "
              f"{format_ci(*list(em['hit@10'].values()))}  "
              f"{format_ci(*list(em['mrr'].values()))}  "
              f"{format_ci(*list(em['ndcg@10'].values()))}")
    
    # Delta
    delta = cp["hit@10"]["mean"] - bm["hit@10"]["mean"]
    print(f"\n  CP vs BM25 Hit@10: {delta*100:+.1f} points")
    print(f"{'─'*70}")
    print(f"Wins: CP={summary['cp_wins']} BM25={summary['bm25_wins']} TIE={summary['ties']}")
    
    return {
        "benchmark": "retrieval",
        "name": name,
        "tier": tier,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }


def get_output_path(name: str) -> Path:
    """Generate timestamped output path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"retrieval_{name}_{timestamp}.json"


async def main():
    parser = argparse.ArgumentParser(
        description="Retrieval Benchmark: File Selection Quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m eval.retrieval.runner --repo flask --no-embeddings
    python -m eval.retrieval.runner --tier tier1 --no-embeddings
    python -m eval.retrieval.runner --tier tier3 --no-embeddings  # private repos
    python -m eval.retrieval.runner --all --no-embeddings
        """
    )
    parser.add_argument("--repo", help="Run single repo (e.g., flask, fastapi)")
    parser.add_argument("--tier", help="Run all repos in tier (tier1, tier2, tier3)")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip slow embeddings baseline")
    parser.add_argument("--output", "-o", help="Override output path")
    args = parser.parse_args()
    
    if not (args.repo or args.tier or args.all):
        parser.print_help()
        return
    
    # Find benchmark files
    benchmark_files = []
    if args.repo:
        # Find by repo name
        for f in QUESTIONS_DIR.rglob("*.json"):
            if f.stem == args.repo:
                benchmark_files.append(f)
                break
    elif args.tier:
        tier_dir = QUESTIONS_DIR / args.tier
        if tier_dir.exists():
            benchmark_files = list(tier_dir.glob("*.json"))
    elif args.all:
        benchmark_files = list(QUESTIONS_DIR.rglob("*.json"))
    
    if not benchmark_files:
        print(f"No benchmark files found")
        return
    
    print(f"Running {len(benchmark_files)} benchmark(s)")
    
    all_results = []
    for bf in sorted(benchmark_files):
        result = await run_benchmark(str(bf), args.no_embeddings)
        result["timestamp"] = datetime.now().isoformat()
        all_results.append(result)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    elif len(all_results) == 1:
        output_path = get_output_path(all_results[0]["name"])
    else:
        name = args.tier if args.tier else "all"
        output_path = get_output_path(name)
    
    # Save results with per-tier aggregation
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        if len(all_results) == 1:
            json.dump(all_results[0], f, indent=2)
        else:
            # Per-tier aggregation (per EVALUATION_STRATEGY.md)
            tier_results = {}
            for r in all_results:
                tier = r.get("tier", "unknown")
                if tier not in tier_results:
                    tier_results[tier] = []
                tier_results[tier].append(r)
            
            def aggregate_tier(results_list):
                """Aggregate metrics across repos in a tier."""
                all_cp_hit10 = []
                all_bm25_hit10 = []
                all_cp_mrr = []
                all_bm25_mrr = []
                
                for r in results_list:
                    # Collect per-question metrics from results
                    for qr in r.get("results", []):
                        all_cp_hit10.append(qr["cp_metrics"]["hit@10"])
                        all_bm25_hit10.append(qr["bm25_metrics"]["hit@10"])
                        all_cp_mrr.append(qr["cp_metrics"]["mrr"])
                        all_bm25_mrr.append(qr["bm25_metrics"]["mrr"])
                
                n = len(all_cp_hit10)
                if n == 0:
                    return {"questions": 0}
                
                cp_hit_mean, cp_hit_lo, cp_hit_hi = bootstrap_ci(all_cp_hit10)
                bm25_hit_mean, bm25_hit_lo, bm25_hit_hi = bootstrap_ci(all_bm25_hit10)
                cp_mrr_mean, cp_mrr_lo, cp_mrr_hi = bootstrap_ci(all_cp_mrr)
                bm25_mrr_mean, bm25_mrr_lo, bm25_mrr_hi = bootstrap_ci(all_bm25_mrr)
                
                return {
                    "questions": n,
                    "repos": len(results_list),
                    "cp_hit@10": {"mean": cp_hit_mean, "ci_lower": cp_hit_lo, "ci_upper": cp_hit_hi},
                    "bm25_hit@10": {"mean": bm25_hit_mean, "ci_lower": bm25_hit_lo, "ci_upper": bm25_hit_hi},
                    "cp_mrr": {"mean": cp_mrr_mean, "ci_lower": cp_mrr_lo, "ci_upper": cp_mrr_hi},
                    "bm25_mrr": {"mean": bm25_mrr_mean, "ci_lower": bm25_mrr_lo, "ci_upper": bm25_mrr_hi},
                    "delta_hit@10": cp_hit_mean - bm25_hit_mean,
                }
            
            tier_summaries = {tier: aggregate_tier(results) for tier, results in tier_results.items()}
            total_q = sum(r["summary"]["questions"] for r in all_results)
            
            # Print per-tier summary
            print(f"\n{'='*70}")
            print(f"AGGREGATE RESULTS BY TIER")
            print(f"{'='*70}")
            for tier in sorted(tier_summaries.keys()):
                ts = tier_summaries[tier]
                if ts["questions"] == 0:
                    continue
                print(f"\n{tier.upper()} ({ts['repos']} repos, {ts['questions']} questions):")
                print(f"  CP Hit@10:   {format_ci(ts['cp_hit@10']['mean'], ts['cp_hit@10']['ci_lower'], ts['cp_hit@10']['ci_upper'])}")
                print(f"  BM25 Hit@10: {format_ci(ts['bm25_hit@10']['mean'], ts['bm25_hit@10']['ci_lower'], ts['bm25_hit@10']['ci_upper'])}")
                print(f"  Delta:       {ts['delta_hit@10']*100:+.1f} points")
            
            json.dump({
                "benchmark": "retrieval",
                "total_questions": total_q,
                "by_tier": tier_summaries,
                "benchmarks": all_results,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
