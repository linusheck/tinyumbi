"""Exercise benchmark workers on real exports with differing payload types."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import index, rational

from evaluation.benchmark import METRICS, benchmark_readers, worker
from evaluation.limits import LimitedRun

FIXTURES = Path(__file__).parent / "fixtures" / "umbi"


@pytest.mark.parametrize("reader,status", [("tinyumbi", "timeout"), ("umbi", "memout")])
def test_reader_limits_are_independent(monkeypatch, reader, status):
    import evaluation.benchmark as benchmark

    real_run = benchmark.run_limited

    def limited_run(command, **kwargs):
        if command[command.index("--worker") + 1] == reader:
            return LimitedRun(-9, "", "", 120, 4097, status)
        return real_run(command, **kwargs)

    monkeypatch.setattr(benchmark, "run_limited", limited_run)
    results = benchmark_readers(FIXTURES / "simple1.umb", repeats=1)
    assert results[reader]["status"] == status
    assert results[reader]["median"] is None
    other = "umbi" if reader == "tinyumbi" else "tinyumbi"
    assert results[other]["status"] == "ok"
    assert results[other]["median"]["total_s"] > 0


@pytest.mark.parametrize("bits", [128, 256, 512])
@pytest.mark.parametrize("interval", [False, True])
@pytest.mark.parametrize("defaults", [False, True])
def test_native_rational_payload(archive, monkeypatch, bits, interval, defaults):
    import tinyumbi

    kind = "rational-interval" if interval else "rational"
    desc = {"type": kind, "size": bits * (2 if interval else 1)}
    members = (
        {}
        if defaults
        else {"branch-to-probability.bin": rational(-1, 3, bits // 16) * (6 if interval else 3)}
    )
    path = archive(index(**{"branch-probability-type": desc}), members)

    def unexpected_scalar(*args):
        pytest.fail("Native rational payload touch must not decode individual scalars")

    with monkeypatch.context() as patch:
        patch.setattr(tinyumbi, "rational_at", unexpected_scalar)
        assert worker("tinyumbi", path)["total_s"] > 0


def test_rational_pair_preserves_encoding(archive):
    import tinyumbi

    path = archive(
        index(**{"branch-probability-type": {"type": "rational"}}),
        {"branch-to-probability.bin": rational(-2, 6) + rational(1, 2) + rational(0, 3)},
    )
    assert worker("tinyumbi", path)["total_s"] > 0
    values = tinyumbi.read(path)["branch_to_probability"]
    assert [tinyumbi.rational_at(values, i) for i in range(3)] == [(-2, 6), (1, 2), (0, 3)]


@pytest.mark.parametrize("path", sorted(FIXTURES.glob("*.umb")), ids=lambda p: p.name)
def test_benchmark_real_export(path):
    results = benchmark_readers(path, repeats=1)
    assert set(results) == {"tinyumbi", "umbi"}
    for result in results.values():
        assert result["status"] == "ok"
        (run,) = result["runs"]
        assert set(run) == set(METRICS)
        assert result["median"] == run
        assert run["total_s"] == pytest.approx(run["load_s"] + run["touch_s"])
        assert run["process_s"] >= run["total_s"] > 0
        assert run["peak_rss_mib"] > 0


def test_benchmark_worker_failure(tmp_path):
    path = tmp_path / "invalid.umb"
    path.write_bytes(b"invalid UMB")
    with pytest.raises(RuntimeError, match="tinyumbi benchmark failed"):
        benchmark_readers(path, repeats=1)


def test_benchmark_cli():
    process = subprocess.run(
        [sys.executable, "-m", "evaluation.benchmark", str(FIXTURES / "simple1.umb"), "--repeat", "1"],
        capture_output=True,
        text=True,
        check=True,
    )
    results = json.loads(process.stdout)
    assert set(results) == {"tinyumbi", "umbi"}
    for result in results.values():
        assert len(result["runs"]) == 1
        assert result["median"]["total_s"] > 0


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["missing.umb"], "UMB file not found"),
        ([str(FIXTURES / "simple1.umb"), "--repeat", "0"], "--repeat must be positive"),
    ],
)
def test_benchmark_cli_invalid_input(arguments, message):
    process = subprocess.run(
        [sys.executable, "-m", "evaluation.benchmark", *arguments],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 2
    assert message in process.stderr


@pytest.mark.parametrize("collect_only", [False, True])
def test_empty_export_benchmark_report(tmp_path, collect_only):
    checkout = tmp_path / "observatory"
    (checkout / "umbtest").mkdir(parents=True)
    (checkout / "umbtest" / "tools.py").touch()
    report = tmp_path / "report.json"
    report.write_text("previous report\n")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_corpus.py",
            "--observatory",
            str(checkout),
            "--benchmark-json",
            str(report),
            *(["--collect-only"] if collect_only else []),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert process.returncode == (0 if collect_only else 1), process.stdout + process.stderr
    if not collect_only:
        assert "No reader benchmarks recorded; report not written" in process.stdout
    assert report.read_text() == "previous report\n"
