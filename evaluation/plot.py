"""Plot reader timings: uv run --with matplotlib python -m evaluation.plot."""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, default=Path(".pytest_cache/d/observatory/benchmarks.json")
    )
    parser.add_argument("--output", type=Path, default=Path("evaluation/reader-scatter.png"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    rows = data["runs"]
    if not rows and not data.get("outcomes"):
        parser.error("The benchmark report contains no exports")

    timeout_ms = data.get("limits", {}).get("timeout_s", 120) * 1000
    times = [
        result["median"]["total_s"] * 1000
        for row in rows
        for result in row["results"].values()
        if result.get("status", "ok") == "ok"
    ]
    lower = 10 ** math.floor(math.log10(min(times))) if times else 1
    upper = max(timeout_ms, max(times, default=1))
    overflow = upper * 2

    def coordinate(result):
        if result.get("status") in {"timeout", "memout"}:
            return overflow
        return result["median"]["total_s"] * 1000

    plt.style.use("default")
    fig, ax = plt.subplots()
    if not rows:
        ax.text(0.5, 0.5, "No exports completed", ha="center", transform=ax.transAxes)
    for exporter in sorted({row["exporter"] for row in rows}):
        group = [row for row in rows if row["exporter"] == exporter]
        x = [coordinate(row["results"]["umbi"]) for row in group]
        y = [coordinate(row["results"]["tinyumbi"]) for row in group]
        ax.scatter(x, y, label=exporter)

    ax.plot([lower, upper], [lower, upper], "k--", label="Equal time")
    ax.axhline(overflow, color="0.5", linestyle=":")
    ax.axvline(overflow, color="0.5", linestyle=":")
    ax.set(
        xscale="log",
        yscale="log",
        xlim=(lower, overflow * 1.4),
        ylim=(lower, overflow * 1.4),
        xlabel="umbi load + touch (ms)",
        ylabel="tinyumbi load + touch (ms)",
        title="tinyumbi vs umbi",
    )
    if data.get("payload", "").startswith("native rational arrays"):
        ax.set_title("tinyumbi vs umbi\nNative rational arrays; umbi eagerly decodes")
    ticks = [10**power for power in range(int(math.log10(lower)), int(math.log10(upper)) + 1)]
    labels = [f"$10^{{{int(math.log10(tick))}}}$" for tick in ticks]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_ticks([*ticks, overflow], [*labels, "TO/MO"])
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
