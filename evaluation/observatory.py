"""Read-only checker for umb-observatory's Tester tool chains."""

import traceback
from pathlib import Path

from umbtest.tools import ReportedResults, UmbTool

import tinyumbi
from evaluation.payload import touch_model


class TinyUmbReader(UmbTool):
    """Read and touch an export, optionally applying a semantic comparison callback."""

    name = "tinyumbi"

    def __init__(self, check_model=None):
        super().__init__()
        self.check_model = check_model

    def check_process(self):
        return True

    def check_umb(self, umb_file, log_file=None, properties=None):
        result = ReportedResults()
        result.logfile = log_file
        if properties:
            result.exit_code = 1
            result.not_supported = True
            result.errors = ("tinyumbi reads models; it does not evaluate temporal properties",)
            return result
        try:
            model = tinyumbi.read(umb_file)
            touch_model(model)
            result.model_info = {
                "states": len(model["state_to_choices"]) - 1,
                "choices": len(model["choice_to_branches"]) - 1,
                "transitions": len(model["branch_to_target"]),
            }
            ts = model["index"]["transition-system"]
            for statistic, entity in (
                ("states", "states"),
                ("choices", "choices"),
                ("transitions", "branches"),
            ):
                if result.model_info[statistic] != ts[f"#{entity}"]:
                    raise ValueError(f"Decoded {statistic} count differs from index.json")
            if self.check_model is not None:
                self.check_model(model, umb_file)
            result.exit_code = 0
            log = f"tinyumbi read and decoded {umb_file}: {result.model_info}\n"
        except Exception as error:
            result.exit_code = 1
            result.errors = (f"{type(error).__name__}: {error}",)
            log = traceback.format_exc()
        if log_file is not None:
            Path(log_file).write_text(log)
        return result
