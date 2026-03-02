#!/usr/bin/env python3
"""
E2E Benchmark: Answer Quality

Measures: Are LLM answers correct when using ContextPacker context?
Method: LLM-as-judge (GPT-4) scores answers against key facts
Compares: CP context vs No context vs Embeddings context

Usage:
    python -m eval.e2e.runner --help
    python -m eval.e2e.runner --questions core --limit 5
    python -m eval.e2e.runner --questions core --approaches contextpacker,no_context
"""

import argparse
import asyncio
import json
import os
import time
import httpx
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

# =============================================================================
# Configuration
# =============================================================================

QUESTIONS_DIR = Path(__file__).parent / "questions"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CP_API_URL = "https://contextpacker.com/v1/packs"
CP_API_KEY = os.getenv("CONTEXTPACKER_API_KEY", "cp_live_your_key_here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "your_gemini_api_key_here")
GITHUB_PAT = os.getenv("GITHUB_PAT", "")  # For private repos

# Cross-vendor setup: OpenAI answers, Gemini judges
ANSWER_MODEL = "gpt-4.1-mini"  # OpenAI cheap/fast
JUDGE_MODEL = "gemini-2.0-flash"  # Google cross-vendor judge (reliable, no thinking overhead)

# Private repo patterns (require GITHUB_PAT)
PRIVATE_REPO_OWNERS = {"rozetyp"}


# =============================================================================
# Context Retrieval
# =============================================================================

def is_private_repo(repo_url: str) -> bool:
    """Check if repo URL belongs to a private repo owner."""
    for owner in PRIVATE_REPO_OWNERS:
        if f"github.com/{owner}/" in repo_url:
            return True
    return False


async def get_embeddings_context(repo_url: str, query: str) -> str:
    """Get context using embeddings-based file search."""
    try:
        from eval.shared.embeddings import search as embeddings_search
        from eval.shared.clone import clone_repo
        
        # Search using embeddings
        files, _ = embeddings_search(repo_url, query, top_k=10, github_pat=GITHUB_PAT)
        if not files:
            return ""
        
        # Clone repo to read file contents
        repo_path = clone_repo(repo_url, GITHUB_PAT)
        if not repo_path:
            return ""
        
        # Build context from top files
        context = f"## Context for: {query}\n*{len(files)} files from embeddings search*\n\n"
        for file_path in files[:5]:  # Top 5 files
            full_path = repo_path / file_path
            if full_path.exists():
                try:
                    content = full_path.read_text(errors="ignore")[:3000]
                    context += f"### {file_path}\n```\n{content}\n```\n\n"
                except:
                    pass
        
        return context[:50000]
    except Exception as e:
        print(f"    ⚠️ Embeddings error: {e}")
        return ""


async def get_cp_context(repo_url: str, query: str) -> str:
    """Get context from ContextPacker."""
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            payload = {"repo_url": repo_url, "query": query, "max_tokens": 30000}
            
            # Add VCS token for private repos
            if is_private_repo(repo_url) and GITHUB_PAT:
                payload["vcs"] = {"provider": "github", "token": GITHUB_PAT}
            
            resp = await client.post(
                CP_API_URL,
                headers={"X-API-Key": CP_API_KEY, "Content-Type": "application/json"},
                json=payload
            )
            if resp.status_code == 200:
                data = resp.json()
                # Context is in markdown field
                context = data.get("markdown", "")
                return context[:50000]
            else:
                print(f"    ⚠️ CP API error: {resp.status_code} - {resp.text[:200]}")
        except Exception as e:
            print(f"    ⚠️ CP API exception: {e}")
    return ""


# =============================================================================
# Answer Generation
# =============================================================================

async def generate_answer(query: str, context: str, approach: str) -> str:
    """Generate answer using GPT-4.1-mini (OpenAI)."""
    if not OPENAI_API_KEY:
        return "[No OPENAI_API_KEY set]"
    
    if approach == "no_context":
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": query}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant. Use the provided code context to answer accurately."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": ANSWER_MODEL, "messages": messages, "max_tokens": 1000}
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            else:
                return f"[API error: {resp.status_code} - {resp.text[:200]}]"
        except Exception as e:
            return f"[Error: {e}]"
    return "[Failed to generate answer]"


# =============================================================================
# LLM-as-Judge Scoring
# =============================================================================

async def score_answer(answer: str, key_facts: List[str], query: str) -> Dict[str, Any]:
    """Score answer against key facts using Gemini 2.5 Pro as cross-vendor judge."""
    if not GEMINI_KEY:
        return {"score": 0, "reasoning": "No GEMINI_API_KEY set"}
    
    facts_text = "\n".join(f"- {f}" for f in key_facts)
    
    prompt = f"""You are an expert code reviewer evaluating an answer about a codebase.

Question: {query}

Key facts that should be covered:
{facts_text}

Answer to evaluate:
{answer}

Score the answer from 0-10 based on:
- Accuracy: Does it correctly describe the code behavior?
- Completeness: Does it cover the key facts?
- Clarity: Is it well-explained?

Respond ONLY with JSON, no other text:
{{"score": <0-10>, "facts_covered": <count>, "reasoning": "<brief explanation>"}}
"""
    
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            # Use Gemini API for cross-vendor judging
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent",
                params={"key": GEMINI_KEY},
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 1000  # Higher for thinking models
                    }
                }
            )
            if resp.status_code == 200:
                data = resp.json()
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                # Handle markdown-wrapped JSON from thinking models
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                result = json.loads(text)
                return result
            else:
                return {"score": 0, "reasoning": f"Gemini API error: {resp.status_code}"}
        except Exception as e:
            return {"score": 0, "reasoning": f"Error: {e}"}
    return {"score": 0, "reasoning": "Failed to score"}


# =============================================================================
# Runner
# =============================================================================

@dataclass
class QuestionResult:
    id: str
    query: str
    repo_url: str
    key_facts: List[str]
    approaches: Dict[str, Dict[str, Any]]  # approach -> {answer, score, reasoning}


async def run_question(question: Dict, approaches: List[str]) -> QuestionResult:
    """Run a single question across all approaches."""
    query = question["query"]
    repo_url = question["repo_url"]
    key_facts = question["key_facts"]
    
    print(f"  [{question['id']}] {query[:50]}...")
    
    results = {}
    
    for approach in approaches:
        # Get context
        if approach == "contextpacker":
            context = await get_cp_context(repo_url, query)
        elif approach == "embeddings":
            context = await get_embeddings_context(repo_url, query)
        elif approach == "no_context":
            context = ""
        else:
            print(f"    ⚠️ Unknown approach: {approach}")
            context = ""
        
        # Generate answer
        answer = await generate_answer(query, context, approach)
        
        # Score answer
        score_result = await score_answer(answer, key_facts, query)
        
        results[approach] = {
            "answer": answer[:500],  # Truncate for storage
            "score": score_result.get("score", 0),
            "facts_covered": score_result.get("facts_covered", 0),
            "reasoning": score_result.get("reasoning", ""),
        }
        
        print(f"    {approach}: score={score_result.get('score', 0)}/10")
    
    return QuestionResult(
        id=question["id"],
        query=query,
        repo_url=repo_url,
        key_facts=key_facts,
        approaches=results
    )


async def run_benchmark(questions_file: str, approaches: List[str], limit: int = None) -> Dict[str, Any]:
    """Run E2E benchmark."""
    with open(questions_file) as f:
        data = json.load(f)
    
    questions = data["questions"][:limit] if limit else data["questions"]
    name = Path(questions_file).stem
    
    print(f"\n{'='*60}")
    print(f"E2E BENCHMARK: {name}")
    print(f"Questions: {len(questions)}")
    print(f"Approaches: {', '.join(approaches)}")
    print(f"{'='*60}\n")
    
    results = []
    for q in questions:
        result = await run_question(q, approaches)
        results.append(result)
    
    # Aggregate scores
    summary = {}
    for approach in approaches:
        scores = [r.approaches[approach]["score"] for r in results]
        summary[approach] = {
            "avg_score": sum(scores) / len(scores),
            "min_score": min(scores),
            "max_score": max(scores),
        }
    
    # Print summary
    print(f"\n{'─'*60}")
    print(f"RESULTS")
    print(f"{'─'*60}")
    for approach, stats in summary.items():
        print(f"  {approach}: avg={stats['avg_score']:.1f}/10 (range: {stats['min_score']}-{stats['max_score']})")
    
    # Determine winner
    if len(approaches) > 1:
        winner = max(summary.keys(), key=lambda a: summary[a]["avg_score"])
        print(f"\n  Winner: {winner}")
    
    return {
        "benchmark": "e2e",
        "name": name,
        "questions": len(questions),
        "approaches": approaches,
        "summary": summary,
        "results": [asdict(r) for r in results],
    }


def get_output_path(name: str) -> Path:
    """Generate timestamped output path."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return RESULTS_DIR / f"e2e_{name}_{timestamp}.json"


async def main():
    parser = argparse.ArgumentParser(
        description="E2E Benchmark: Answer Quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m eval.e2e.runner --questions core --limit 3
    python -m eval.e2e.runner --questions core --approaches contextpacker,no_context
        """
    )
    parser.add_argument("--questions", required=True, help="Question file name (e.g., core)")
    parser.add_argument("--approaches", default="contextpacker,no_context", 
                        help="Comma-separated approaches to compare")
    parser.add_argument("--limit", type=int, help="Limit number of questions")
    parser.add_argument("--output", "-o", help="Override output path")
    args = parser.parse_args()
    
    # Find questions file
    questions_file = QUESTIONS_DIR / f"{args.questions}.json"
    if not questions_file.exists():
        print(f"Questions file not found: {questions_file}")
        return
    
    approaches = [a.strip() for a in args.approaches.split(",")]
    
    # Check API key
    if not OPENAI_API_KEY:
        print("⚠️  OPENAI_API_KEY not set - answers will fail")
    
    result = await run_benchmark(str(questions_file), approaches, args.limit)
    result["timestamp"] = datetime.now().isoformat()
    
    # Save results
    output_path = Path(args.output) if args.output else get_output_path(args.questions)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
