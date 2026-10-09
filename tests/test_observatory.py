"""Run the upstream model corpus through real exporters and the read-only checker."""

import json
import os
import sys
from pathlib import Path

import pytest
import umbi.umb
from conftest import BENCHMARK_OUTCOMES, BENCHMARK_ROWS
from test_reader import assert_reference_matches

from evaluation.benchmark import benchmark_readers
from evaluation.corpus import paper_models
from evaluation.limits import run_limited

pytestmark = pytest.mark.observatory


def pytest_generate_tests(metafunc):
    if "benchmark" not in metafunc.fixturenames:
        return
    if not metafunc.config.getoption("--observatory"):
        metafunc.parametrize(
            "benchmark", [pytest.param(None, marks=pytest.mark.skip(reason="Enable with --observatory PATH"))]
        )
        return
    from umbtest import benchmarks

    if directory := metafunc.config.getoption("--benchmark-models"):
        models = [benchmarks.UmbBenchmark(path) for path in paper_models(directory)]
    else:
        models = (
            benchmarks.prism_files
            if os.environ.get("UMB_TEST_MODELS") == "full"
            else benchmarks.quick_prism_files
        )
    if not models:
        raise pytest.UsageError("The observatory model corpus is empty")
    metafunc.parametrize("benchmark", models, ids=lambda b: str(b.id))


@pytest.fixture(scope="module", params=["storm", "storm-exact", "prism", "prism-exact"])
def exporter(request):
    return request.param


def make_exporter(name):
    from umbtest.tools import PrismCLI, StormCLI, configure_umbtools

    configure_umbtools()
    factories = {
        "storm": lambda: StormCLI(),
        "storm-exact": lambda: StormCLI(extra_args=["--exact"]),
        "prism": lambda: PrismCLI(),
        "prism-exact": lambda: PrismCLI(extra_args=["-exact"]),
    }
    tool = factories[name]()
    try:
        available = tool.check_process()
    except (OSError, RuntimeError):
        available = False
    if not available:
        pytest.skip(f"Observatory exporter {name} unavailable; configure UMB_STORM / UMB_PRISM")
    return tool


def compare_reference(model, path):
    assert_reference_matches(model, umbi.umb.read(path))


def run_export(exporter, benchmark, tmp_path):
    import tinyumbi

    path = tmp_path / "model.umb"
    loader = exporter.prism_file_to_umb(benchmark.location, path, tmp_path / "export.log")
    if loader.not_supported:
        pytest.skip("Exporter does not support this model")
    if loader.anticipated_error:
        pytest.xfail("Upstream marks this exporter error as anticipated")
    assert loader.exit_code == 0, (tmp_path / "export.log").read_text()
    assert path.is_file() and path.stat().st_size > 0
    model = tinyumbi.read(path)
    counts = model["index"]["transition-system"]
    return {
        "model": str(benchmark.id),
        "states": counts["#states"],
        "choices": counts["#choices"],
        "branches": counts["#branches"],
        "archive_bytes": path.stat().st_size,
    }


def test_export_read(exporter, benchmark, tmp_path, request):
    config = request.config
    result_path = tmp_path / "result.json"
    settings = {
        "observatory": str(Path(config.getoption("--observatory")).resolve()),
        "exporter": exporter,
        "model": str(benchmark.location.resolve()),
        "directory": str(tmp_path),
        "result": str(result_path),
        "stage": "export",
    }
    limits = {
        "timeout_s": config.getoption("--benchmark-timeout"),
        "memory_mib": config.getoption("--benchmark-memory-mib"),
    }

    def stage(name):
        settings["stage"] = name
        result_path.unlink(missing_ok=True)
        process = run_limited(
            [sys.executable, str(Path(__file__).with_name("observatory_worker.py")), json.dumps(settings)],
            cwd=Path(__file__).resolve().parents[1],
            **limits,
        )
        if process.limit:
            return {"status": process.limit, "reason": f"{name} exceeded {process.limit} limit"}
        if process.returncode:
            return {"status": "failed", "reason": process.stdout + process.stderr}
        return json.loads(result_path.read_text())

    outcome = {"model": str(benchmark.id), "exporter": exporter, "stage": "export"}
    try:
        result = stage("export")
        outcome.update({key: value for key, value in result.items() if key != "row"})
        config.stash[BENCHMARK_OUTCOMES].append(outcome)
        if result["status"] == "ok":
            row = {**result["row"], "exporter": exporter}
            row["results"] = benchmark_readers(
                tmp_path / "model.umb", config.getoption("--benchmark-repeat"), **limits
            )
            config.stash[BENCHMARK_ROWS].append(row)
            for reader, metrics in row["results"].items():
                config.stash[BENCHMARK_OUTCOMES].append(
                    {
                        "model": str(benchmark.id),
                        "exporter": exporter,
                        "stage": "reader",
                        "reader": reader,
                        "status": metrics["status"],
                    }
                )
            if all(metrics["status"] == "ok" for metrics in row["results"].values()):
                result = stage("compare")
                row["comparison"] = result["status"]
                config.stash[BENCHMARK_OUTCOMES].append(
                    {
                        "model": str(benchmark.id),
                        "exporter": exporter,
                        "stage": "compare",
                        **result,
                    }
                )
            else:
                row["comparison"] = "not_run: reader resource limit"
                return
        if result["status"] == "xfail":
            pytest.xfail(result["reason"])
        elif result["status"] in {"timeout", "memout", "skipped"}:
            if config.getoption("--require-observatory-tools") and "unavailable" in result["reason"]:
                pytest.fail(result["reason"])
            pytest.skip(result["reason"])
        elif result["status"] != "ok":
            pytest.fail(result["reason"])
    finally:
        for path in tmp_path.glob("*.umb"):
            path.unlink()
