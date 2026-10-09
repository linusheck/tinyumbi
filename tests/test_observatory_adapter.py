"""Failure reporting and lazy decoding at the observatory boundary."""

import pytest
from conftest import index, u64

pytest.importorskip("umbtest.tools", reason="Enable with --observatory PATH")

from evaluation.observatory import TinyUmbReader  # noqa: E402


def test_read_only_checker(archive, tmp_path):
    path = archive(index())
    log = tmp_path / "reader.log"
    result = TinyUmbReader().check_umb(path, log_file=log)
    assert result.exit_code == 0
    assert result.model_info == {"states": 3, "choices": 3, "transitions": 3}
    assert "decoded" in log.read_text()


def test_lazy_decode_failure_is_reported(archive, tmp_path):
    path = archive(
        index(**{"#choice-actions": 1}),
        {
            "actions/choices/string-mapping.bin": u64([0, 1]),
            "actions/choices/strings.bin": b"\xff",
        },
    )
    log = tmp_path / "reader.log"
    result = TinyUmbReader().check_umb(path, log_file=log)
    assert result.exit_code == 1
    assert not result.not_supported
    assert not result.anticipated_error
    assert "UnicodeDecodeError" in result.errors[0]
    assert "UnicodeDecodeError" in log.read_text()


def test_reference_mismatch_is_reported(archive):
    def mismatch(model, path):
        raise AssertionError("reference mismatch")

    result = TinyUmbReader(check_model=mismatch).check_umb(archive(index()))
    assert result.exit_code == 1
    assert result.errors == ("AssertionError: reference mismatch",)


def test_temporal_properties_are_unsupported(archive):
    result = TinyUmbReader().check_umb(archive(index()), properties=['P=? [F "done"]'])
    assert result.exit_code == 1
    assert result.not_supported
