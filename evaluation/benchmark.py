"""Benchmark both readers on an existing UMB export in fresh processes."""

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

from evaluation.limits import run_limited

METRICS = ("load_s", "touch_s", "total_s", "process_s", "peak_rss_mib")


def _touch_reference(value):
    # ExplicitUmb already decodes rationals, strings, and valuations during read.
    if isinstance(value, dict):
        for item in value.values():
            _touch_reference(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _touch_reference(item)


def worker(reader, path):
    if reader == "tinyumbi":
        import tinyumbi
        from evaluation.payload import touch_model

        read, touch = tinyumbi.read, touch_model
    else:
        import umbi.umb

        read = umbi.umb.read

        def touch(model):
            for key, value in vars(model).items():
                if key != "index":
                    _touch_reference(value)

    start = time.perf_counter()
    model = read(path)
    loaded = time.perf_counter()
    touch(model)
    end = time.perf_counter()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "load_s": loaded - start,
        "touch_s": end - loaded,
        "total_s": end - start,
        "peak_rss_mib": rss / (1024**2 if sys.platform == "darwin" else 1024),
    }


def benchmark_readers(path, repeats=3, *, timeout_s=120, memory_mib=4096):
    if repeats < 1:
        raise ValueError("Benchmark repeats must be positive")
    results = {}
    for reader in ("tinyumbi", "umbi"):
        runs = []
        for _ in range(repeats):
            process = run_limited(
                [
                    sys.executable,
                    "-m",
                    "evaluation.benchmark",
                    "--worker",
                    reader,
                    str(Path(path).resolve()),
                ],
                cwd=Path(__file__).resolve().parents[1],
                timeout_s=timeout_s,
                memory_mib=memory_mib,
            )
            if process.limit:
                results[reader] = {
                    "status": process.limit,
                    "runs": runs,
                    "median": None,
                    "elapsed_s": process.elapsed_s,
                    "peak_rss_mib": process.peak_rss_mib,
                }
                break
            if process.returncode:
                raise RuntimeError(f"{reader} benchmark failed for {path}:\n{process.stderr}")
            run = json.loads(process.stdout)
            run["process_s"] = process.elapsed_s
            runs.append(run)
        else:
            results[reader] = {
                "status": "ok",
                "runs": runs,
                "median": {key: statistics.median(run[key] for run in runs) for key in METRICS},
            }
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--repeat", type=int, default=3, help="Fresh processes per reader (default: 3)")
    parser.add_argument(
        "--timeout", type=float, default=120, help="Seconds per reader process (default: 120)"
    )
    parser.add_argument(
        "--memory-mib", type=float, default=4096, help="Reader RSS limit in MiB (default: 4096)"
    )
    parser.add_argument("--worker", choices=["tinyumbi", "umbi"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.path.is_file():
        parser.error(f"UMB file not found: {args.path}")
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    if args.timeout <= 0 or args.memory_mib <= 0:
        parser.error("--timeout and --memory-mib must be positive")
    if args.worker:
        print(json.dumps(worker(args.worker, args.path)))
        return
    try:
        results = benchmark_readers(
            args.path, args.repeat, timeout_s=args.timeout, memory_mib=args.memory_mib
        )
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
