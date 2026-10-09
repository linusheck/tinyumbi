# tinyumbi

A compact, read-only, vibe-coded, memory-mapped UMB 1.0 parser. Assumes valid inputs. It's about 300 lines of Python code.

```sh
uv sync --locked
```

```python
import tinyumbi as umb

model = umb.read("model.umb")  # Also accepts gzip, xz, and bzip2 archives.
targets = model["branch_to_target"]
probabilities = model["branch_to_probability"]

# Other helpers:
# umb.valuation_at(model["valuations"]["states"], i)
# umb.column(model["valuations"]["states"], "variable")
# umb.string_at(string_table, i)
# umb.read_archive("model.umb") -> {name: memoryview}
```

Supports the [official UMB 1.0 specification](https://github.com/pmc-tools/umb/blob/main/spec.md) (version dated 2026-05-03).

## Development

```sh
uv run pytest
uv run ruff check src tests evaluation
```

## Benchmarks

Compare readers on an existing export:

```sh
uv run python -m evaluation.benchmark model.umb --repeat 3
```

CI benchmarks all 36 paper models, with a separate 120-second timeout and
4 GiB RSS limit per reader repetition. Its `paper-benchmarks` artifact contains
`paper-benchmarks.json`, `reader-scatter.png`, and `reader-scatter.svg`.
