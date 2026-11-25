# Contributing to evc-landing-estimator

Pull requests are welcome from both human contributors and AI-assisted workflows. Check issues labeled `good first issue` or `help wanted` for scoped tasks, or open an issue first before starting a large change.

## Local setup and tests

All tests and the synthetic demo run locally with no Google Ads, GA4, or CRM credentials required.

```bash
pip install -e ".[dev]"
pytest -q
evc-landing-estimator demo --out-dir out/
```

## What makes a PR easy to merge

- Keep each PR tied to one issue with a small, focused diff.
- Include deterministic unit tests using small synthetic CSV/XLSX fixtures (never real client or company exports).
- Preserve the core modeling guardrails: no revenue/ROAS claims, no silent coercions that hide bad input data, and keep `channel_rules.csv` deterministic with zero LLM calls.
- When adding a new export adapter or format fix in `adapters.py`, add a regression test in `tests/test_adapters.py`.
- Note what changed and include your `pytest -q` result in the PR description.

## Using AI coding agents

Coding agents are welcome on this repo. Just keep changes strictly within the target files for the issue and make sure every new behavior is covered by a test rather than refactoring unrelated modules.
