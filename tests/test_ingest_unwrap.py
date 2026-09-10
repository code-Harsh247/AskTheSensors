"""ats.ingest._unwrap_same_name_nesting, isolated: Kaggle's dataset
processing sometimes materializes a single file as a directory of the same
name containing just that one file (confirmed empirically -- see
tests/test_oracle.py's data_dir_extracted fixture)."""

from __future__ import annotations

from ats.ingest import _unwrap_same_name_nesting


def test_unwraps_one_level_of_same_name_nesting(tmp_path):
    outer = tmp_path / "file.csv"
    outer.mkdir()
    inner = outer / "file.csv"
    inner.write_text("real content")

    resolved = _unwrap_same_name_nesting(outer)
    assert resolved == inner
    assert resolved.read_text() == "real content"


def test_unwraps_two_levels_of_same_name_nesting(tmp_path):
    level0 = tmp_path / "file.csv"
    level1 = level0 / "file.csv"
    level0.mkdir()
    level1.mkdir()
    leaf = level1 / "file.csv"
    leaf.write_text("real content")

    assert _unwrap_same_name_nesting(level0) == leaf


def test_plain_file_is_returned_unchanged(tmp_path):
    plain = tmp_path / "file.csv"
    plain.write_text("real content")
    assert _unwrap_same_name_nesting(plain) == plain


def test_nonexistent_path_is_returned_unchanged(tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    assert _unwrap_same_name_nesting(missing) == missing


def test_directory_with_differently_named_contents_stops_unwrapping(tmp_path):
    """The raw_acc/proc_gyro double-nesting case (a directory containing
    many differently-named entries) must not be mistaken for same-name
    nesting -- _iter_bursts handles that case separately."""
    outer = tmp_path / "raw_acc"
    outer.mkdir()
    (outer / "some_other_name").mkdir()
    assert _unwrap_same_name_nesting(outer) == outer
