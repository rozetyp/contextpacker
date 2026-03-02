# ContextPacker

**Repo-native context retrieval for AI code tools.**

Transforms `(question, repo_url)` → relevant source files, packed within your token budget.
Uses LLM-based file selection instead of embeddings — better precision, no vector index to maintain.

> **Try it now → [contextpacker.com](https://contextpacker.com)** — 100 free requests, no card required.
> We're actively developing this and would love your feedback. Open an issue, drop a ⭐, or reach out.

```
question: "How does routing work?"
repo:     https://github.com/pallets/flask
    ↓
ContextPacker selects 6–10 relevant files
    ↓
Returns packed Markdown ready for your LLM prompt
```

**Benchmarks** (NDCG@10 vs industry baselines):

| System | NDCG@10 | MRR | Hit@10 |
|--------|---------|-----|--------|
| ContextPacker | **0.92** | **0.89** | **98%** |
| Embeddings | 0.79 | 0.74 | 87% |
| BM25 | 0.58 | 0.51 | 74% |

*Evaluated on 47 questions across 5 private repos (zero LLM prior).*

---

## Hosted API

The fastest way to get started: **[contextpacker.com](https://contextpacker.com)**

- 100 free requests, no card required — no credit card, no waitlist
- Works with any GitHub repo (public or private)
- Use our [MCP server](https://github.com/rozetyp/contextpacker-mcp) to connect any MCP-compatible AI editor (Cursor, Claude Desktop, Windsurf, VS Code)

We're in active development. If you try it and something doesn't work, please [open an issue](https://github.com/rozetyp/contextpacker/issues) — it genuinely helps.

---

## Self-hosting

### Prerequisites

- Python 3.11+
- Git
- An LLM API key (Gemini Flash recommended — cheap and fast)

### Setup

```bash
git clone https://github.com/rozetyp/contextpacker
cd contextpacker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: set LLM_API_KEY and API_AUTH_TOKEN
```

### Run

```bash
uvicorn context_packer.main:app --reload --port 8000
```

### Test

```bash
curl localhost:8000/v1/packs \
  -H "X-API-Key: your-token" \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/pallets/flask", "query": "How does routing work?"}'
```

---

## How it works

```
Request
    ↓
Clone repo (shallow, depth=1) + cache
    ↓
Build file tree, sorted by priority (src/ first, tests/ last)
    ↓
Extract AST symbols per file (functions, classes, docstrings)
    ↓
LLM selects most relevant files for the query
    ↓
Pack selected files into Markdown within token budget
    ↓
Return with per-file reason comments
```

Key design choices:
- **File paths carry semantic info** — the LLM sees enriched paths like `src/auth/login.py [login_user, verify_token]`
- **Priority sorting** — source files rank above tests/docs before the LLM even sees the tree
- **Token-budget binary search** — large files are truncated, never skipped
- **Cache** — first call clones + indexes; subsequent calls are ~1s

---

## API reference

### `POST /v1/packs`

Select and pack relevant files.

```json
{
  "repo_url": "https://github.com/pallets/flask",
  "query": "How does routing work?",
  "max_tokens": 8000
}
```

Response:
```json
{
  "markdown": "...",
  "stats": {
    "files_selected": 8,
    "tokens_packed": 7234,
    "tokens_raw_repo": 284000,
    "cache_hit": true
  }
}
```

### `POST /v1/skeleton`

Return the full annotated file tree without file contents.

```json
{
  "repo_url": "https://github.com/pallets/flask",
  "query": "repository overview"
}
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `LLM_API_KEY` | Yes | Gemini / OpenAI / Anthropic key for file selector |
| `API_AUTH_TOKEN` | Yes* | Single auth token (dev/simple setup) |
| `API_KEYS` | Yes* | Multiple keys with tiers: `key:tier:user_id,...` |
| `GITHUB_PAT` | Recommended | Avoids GitHub rate limits on public repos; required for private repos |
| `DATABASE_URL` | No | Postgres URL — if unset, falls back to in-memory |
| `LLM_MODEL` | No | Default: `gemini-2.5-flash` |

*Either `API_AUTH_TOKEN` or `API_KEYS` is required.

---

## Running benchmarks

Make sure your `.env` is set up first (see [Setup](#setup) above) — tests import the app config which requires `LLM_API_KEY`.

```bash
# Unit tests (fast, no API calls)
pytest tests/ -v

# Retrieval benchmarks (~10 min, requires a hosted API key)
export CONTEXTPACKER_API_KEY=cp_live_your_key
python -m eval.retrieval.runner --all --no-embeddings

# Single repo
python -m eval.retrieval.runner --repo flask --no-embeddings
```

---

## Feedback & community

This project is early and actively improving. If you:
- Hit a bug or unexpected result → [open an issue](https://github.com/rozetyp/contextpacker/issues)
- Have a feature idea → open an issue with the `enhancement` label
- Want to share a benchmark result or comparison → we'd love to see it

A ⭐ on GitHub helps more people find the project.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT — see [LICENSE](LICENSE).
