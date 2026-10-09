"""Export or compare one model inside the parent's process-group limits."""

import json
import sys
from pathlib import Path

import pytest

settings = json.loads(sys.argv[1])
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, settings["observatory"])

from test_observatory import compare_reference, make_exporter, run_export  # noqa: E402
from umbtest.benchmarks import UmbBenchmark  # noqa: E402

try:
    if settings["stage"] == "export":
        row = run_export(
            make_exporter(settings["exporter"]),
            UmbBenchmark(Path(settings["model"])),
            Path(settings["directory"]),
        )
        result = {"status": "ok", "row": row}
    else:
        from evaluation.observatory import TinyUmbReader

        checked = TinyUmbReader(check_model=compare_reference).check_umb(
            Path(settings["directory"]) / "model.umb",
            log_file=Path(settings["directory"]) / "comparison.log",
        )
        assert checked.exit_code == 0, checked.errors
        result = {"status": "ok"}
except pytest.skip.Exception as error:
    result = {"status": "skipped", "reason": str(error)}
except pytest.xfail.Exception as error:
    result = {"status": "xfail", "reason": str(error)}
Path(settings["result"]).write_text(json.dumps(result))
