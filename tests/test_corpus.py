"""Discover the paper layout without depending on the observatory checkout."""

import pytest

from evaluation.corpus import paper_models


def test_paper_model_discovery(tmp_path):
    for name in ("wlan.4-0", "cluster.64-2000-20"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "model.prism").write_text("mdp\n")
        (directory / "model.jani").write_text("{}\n")
    (tmp_path / "README.md").write_text("Corpus documentation\n")
    assert [path.parent.name for path in paper_models(tmp_path)] == ["cluster.64-2000-20", "wlan.4-0"]


def test_empty_paper_corpus_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="expected .*/model.prism"):
        paper_models(tmp_path)
