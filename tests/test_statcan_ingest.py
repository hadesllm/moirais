# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for morie.ingest.statcan — the StatCan direct-download ingest.

The live download (a ~567 MB StatCan PUMF zip) is not exercised here;
these tests cover the zip/CSV extraction logic with synthetic archives
and confirm the catalog wiring.
"""
import zipfile

import pytest

from morie.ingest.statcan import _csv_from_zip


def _make_zip(tmp_path, files):
    p = tmp_path / "statcan.zip"
    with zipfile.ZipFile(p, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return str(p)


def test_csv_from_zip_reads_first_csv(tmp_path):
    z = _make_zip(tmp_path, {"data.csv": "a,b\n1,2\n3,4\n"})
    df = _csv_from_zip(z)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_csv_from_zip_named_member(tmp_path):
    z = _make_zip(tmp_path, {
        "readme.txt": "ignore me",
        "first.csv": "x\n1\n",
        "second.csv": "y,z\n9,8\n",
    })
    df = _csv_from_zip(z, member="second.csv")
    assert list(df.columns) == ["y", "z"]


def test_csv_from_zip_no_csv_raises(tmp_path):
    z = _make_zip(tmp_path, {"readme.txt": "no data here"})
    with pytest.raises(ValueError, match="no .csv"):
        _csv_from_zip(z)


def test_csv_from_zip_bad_member_raises(tmp_path):
    z = _make_zip(tmp_path, {"data.csv": "a\n1\n"})
    with pytest.raises(KeyError, match="not in the archive"):
        _csv_from_zip(z, member="missing.csv")


def test_cchs22_registered_in_catalog():
    from morie.data import DATASET_CATALOG
    entry = DATASET_CATALOG["cchs22"]
    assert entry["fetcher"] == "morie.ingest.statcan:fetch_statcan_csv"
    assert entry["fetcher_args"]["url"].endswith("2022_CSV.zip")
    assert entry["source"] == "statcan"
