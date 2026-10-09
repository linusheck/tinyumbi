import io
import json
import os
import platform
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from evaluation.corpus import paper_models

BENCHMARK_ROWS = pytest.StashKey[list]()
BENCHMARK_OUTCOMES = pytest.StashKey[list]()


def pytest_addoption(parser):
    parser.addoption(
        "--benchmark-timeout",
        type=float,
        default=120,
        help="Wall-clock seconds per reader repetition (export and comparison bounded separately)",
    )
    parser.addoption(
        "--benchmark-memory-mib",
        type=float,
        default=4096,
        help="RSS limit in MiB per reader process group",
    )
    parser.addoption(
        "--benchmark-models",
        type=Path,
        default=os.environ.get("UMB_BENCHMARK_MODELS"),
        help="Paper model directory containing */model.prism; replaces the observatory corpus",
    )
    parser.addoption(
        "--observatory",
        default=os.environ.get("UMB_OBSERVATORY"),
        help="Path to an umb-observatory checkout; enables exporter interoperability tests",
    )
    parser.addoption(
        "--require-observatory-tools",
        action="store_true",
        help="Fail if a selected observatory exporter is unavailable",
    )
    parser.addoption(
        "--benchmark-repeat",
        type=int,
        default=3,
        help="Fresh processes per reader for each observatory export (default: 3)",
    )
    parser.addoption(
        "--benchmark-json",
        type=Path,
        help="Save observatory sizes and reader benchmarks here (default: pytest cache)",
    )


def pytest_configure(config):
    config.stash[BENCHMARK_ROWS] = []
    config.stash[BENCHMARK_OUTCOMES] = []
    for option in ("--benchmark-timeout", "--benchmark-memory-mib"):
        if config.getoption(option) <= 0:
            raise pytest.UsageError(f"{option} must be positive")
    if config.getoption("--benchmark-repeat") < 1:
        raise pytest.UsageError("--benchmark-repeat must be positive")
    if config.getoption("--require-observatory-tools") and not config.getoption("--observatory"):
        raise pytest.UsageError("--require-observatory-tools requires --observatory")
    if directory := config.getoption("--benchmark-models"):
        if not config.getoption("--observatory"):
            raise pytest.UsageError("--benchmark-models requires --observatory for the exporter adapters")
        try:
            paper_models(directory)
        except ValueError as error:
            raise pytest.UsageError(str(error)) from error
    if config.getoption("--benchmark-json") and not config.getoption("--observatory"):
        raise pytest.UsageError("--benchmark-json requires --observatory; it reports exported-model timings")
    if checkout := config.getoption("--observatory"):
        root = Path(checkout).resolve()
        if not (root / "umbtest" / "tools.py").is_file():
            raise pytest.UsageError(f"Not an umb-observatory checkout: {root}")
        sys.path.insert(0, str(root))


def pytest_sessionfinish(session):
    config = session.config
    rows = config.stash[BENCHMARK_ROWS]
    if not rows and not config.stash[BENCHMARK_OUTCOMES]:
        if (
            (config.getoption("--benchmark-models") or config.getoption("--benchmark-json"))
            and not config.option.collectonly
            and session.exitstatus == pytest.ExitCode.OK
        ):
            reporter = config.pluginmanager.getplugin("terminalreporter")
            if reporter is not None:
                reporter.write_sep(
                    "=",
                    "No reader benchmarks recorded; report not written. "
                    "Check skips, UMB_STORM / UMB_PRISM, and the -k / -m selection.",
                    red=True,
                )
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        return
    path = config.getoption("--benchmark-json") or config.cache.mkdir("observatory") / "benchmarks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "environment": {"platform": platform.platform(), "python": platform.python_version()},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "cache": "warm filesystem cache; independent worker processes after export",
                "payload": "native rational arrays; strings and valuation rows decoded; umbi eagerly decodes",
                "repeats": config.getoption("--benchmark-repeat"),
                "corpus": "paper" if config.getoption("--benchmark-models") else "observatory",
                "selection": config.option.keyword,
                "limits": {
                    "timeout_s": config.getoption("--benchmark-timeout"),
                    "memory_mib": config.getoption("--benchmark-memory-mib"),
                    "scope": "per reader repetition; export and comparison bounded separately",
                    "memory": "aggregate RSS sampled every 100 ms",
                },
                "outcomes": config.stash[BENCHMARK_OUTCOMES],
                "runs": rows,
            },
            indent=2,
        )
        + "\n"
    )


def pytest_terminal_summary(terminalreporter):
    config = terminalreporter.config
    outcomes = config.stash[BENCHMARK_OUTCOMES]
    if outcomes:
        counts = Counter(outcome["status"] for outcome in outcomes)
        terminalreporter.write_line(
            "Model/exporter outcomes: " + ", ".join(f"{k}={v}" for k, v in counts.items())
        )
    rows = config.stash[BENCHMARK_ROWS]
    if not rows:
        return
    terminalreporter.section("Reader benchmarks (largest 10 exports by states)")
    terminalreporter.write_line(
        "Median load + payload touch (native rational arrays); "
        "imports excluded; peak RSS from fresh processes."
    )
    terminalreporter.write_line(
        f"{'Model / exporter':<48} {'States':>8} {'Choices':>8} {'Branches':>9} {'KiB':>8} "
        f"{'tiny ms':>9} {'umbi ms':>9} {'tiny MiB':>9} {'umbi MiB':>9}"
    )

    def metrics(result):
        if result["status"] != "ok":
            return result["status"], f"{result['peak_rss_mib']:.1f}"
        median = result["median"]
        return f"{median['total_s'] * 1000:.2f}", f"{median['peak_rss_mib']:.1f}"

    for row in sorted(rows, key=lambda r: r["states"], reverse=True)[:10]:
        tiny_time, tiny_memory = metrics(row["results"]["tinyumbi"])
        ref_time, ref_memory = metrics(row["results"]["umbi"])
        model_name = row["model"].removesuffix("/model.prism").split("/")[-1]
        terminalreporter.write_line(
            f"{model_name + ' / ' + row['exporter']:<48} "
            f"{row['states']:>8,} {row['choices']:>8,} {row['branches']:>9,} "
            f"{row['archive_bytes'] / 1024:>8.1f} "
            f"{tiny_time:>9} {ref_time:>9} {tiny_memory:>9} {ref_memory:>9}"
        )
    terminalreporter.write_line(
        f"{len(rows)} exports from {len({row['model'] for row in rows})} source models; "
        f"states {min(r['states'] for r in rows):,}–{max(r['states'] for r in rows):,}; "
        f"branches {min(r['branches'] for r in rows):,}–{max(r['branches'] for r in rows):,}."
    )
    path = config.getoption("--benchmark-json") or config.cache.mkdir("observatory") / "benchmarks.json"
    terminalreporter.write_line(f"Full sizes, individual runs, and medians: {path}")


def index(n=3, **changes):
    ts = {
        "time": "discrete",
        "#players": 1,
        "#states": n,
        "#initial-states": 0,
        "#choices": n,
        "#choice-actions": 0,
        "#branches": n,
        "#branch-actions": 0,
        "#observations": 0,
    }
    ts.update(changes)
    return {"format-version": 1, "format-revision": 0, "transition-system": ts}


def u64(values):
    return np.asarray(values, dtype="<u8").tobytes()


def u32(values):
    return np.asarray(values, dtype="<u4").tobytes()


def doubles(values):
    return np.asarray(values, dtype="<f8").tobytes()


def rational(numerator, denominator, width=8):
    return numerator.to_bytes(width, "little", signed=True) + denominator.to_bytes(width, "little")


@pytest.fixture
def archive(tmp_path):
    def make(idx, members=None, compression="", prefix="", index_first=True):
        path = tmp_path / f"model-{compression or 'tar'}.umb"
        items = {"index.json": json.dumps(idx).encode(), **(members or {})}
        if not index_first:
            items["index.json"] = items.pop("index.json")
        with tarfile.open(path, f"w:{compression}", format=tarfile.PAX_FORMAT) as tar:
            for name, data in items.items():
                info = tarfile.TarInfo(prefix + name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return path

    return make
