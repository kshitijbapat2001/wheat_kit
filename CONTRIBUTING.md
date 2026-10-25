# Contributing to wheatkit

Thank you for helping. Maintainers aim to reply to every issue and PR within
**48 hours**. If we're slower than that, ping us.

## Ways to contribute (no GPU needed)

| Contribution | Where | What we check |
|---|---|---|
| **Add or correct a dataset** | `src/wheatkit/data/registry.yaml` | Licence taken from the *original* record (link it in the PR), tier and reason, md5 for Zenodo files |
| **Add a decision rule** | `src/wheatkit/decide/rules.yaml` | At least one citation, region scope, growth-stage window, **no products, active ingredients or rates** (enforced by tests) |
| **Add a metric, loader or trait** | `src/wheatkit/...` + `tests/` | Tiny synthetic fixture test that runs on CPU in under 2 s |
| **Submit a benchmark result** | `benchmarks/` | Predictions file + model card; computed on the official split |
| **Docs and notebooks** | `docs/`, `notebooks/` | Runs top to bottom on Colab |

Good first issues are labelled
[`good first issue`](https://github.com/kshitijbapat2001/wheatkit/labels/good%20first%20issue).

## Development setup

```bash
git clone https://github.com/kshitijbapat2001/wheatkit && cd wheatkit
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,torch]"
pre-commit install
pytest -m "not slow"          # under 5 s
pytest                        # includes a 1-step CPU training smoke test
```

## Ground rules

1. **Honest evaluation.** Report results on official or group splits, with at least 3 seeds and per-domain tables. Random splits must be labelled as diagnostics.
2. **Licences travel with data.** Never commit dataset files or weights to the repo. Never re-host data; point to the original record.
3. **Optional heavy dependencies** go behind extras (`[hf]`, `[geo]`, `[ultralytics]`). The core must import without torch.
4. **No AGPL code in the core.** Ultralytics integrations live behind the `[ultralytics]` extra, and their outputs are labelled AGPL.
5. Style is enforced by `black` and `flake8` (via `pre-commit`). Use type hints on public functions and Google- or NumPy-style docstrings.

## Commit and PR conventions

- Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`, `data:`, `rule:`, `bench:`).
- Keep each PR to one logical change, update `CHANGELOG.md` under *Unreleased*, and link the issue.

By contributing you agree that your contribution is licensed under Apache-2.0 and that
you will follow the [Code of Conduct](CODE_OF_CONDUCT.md).
