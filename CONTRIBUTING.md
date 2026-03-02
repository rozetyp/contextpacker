# Contributing

Contributions, bug reports, and feature requests are welcome!

## Getting started

```bash
git clone https://github.com/rozetyp/contextpacker
cd contextpacker
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in LLM_API_KEY
```

## Running tests

```bash
pytest tests/ -v
```

## Running retrieval benchmarks

```bash
export CONTEXTPACKER_API_KEY=cp_live_your_key
python -m eval.retrieval.runner --repo flask
```

## Pull requests

- Open an issue first for larger changes
- Keep PRs focused and small
- Add tests for new functionality
