"""Tests for morie.fn.swalg."""
import numpy as np
from morie.fn.swalg import smith_waterman


def test_swalg_identical():
    result = smith_waterman(seq1="ACGTACGT", seq2="ACGTACGT")
    assert isinstance(result.value, int)
    assert result.value == 16
    assert result.extra["alignment1"] == "ACGTACGT"
    assert result.extra["alignment2"] == "ACGTACGT"
    assert result.extra["length"] == 8


def test_swalg_partial_match():
    result = smith_waterman(seq1="ACGT", seq2="XXACGTXX")
    assert isinstance(result.value, int)
    assert result.value > 0
    assert isinstance(result.name, str) and len(result.name) > 0


def test_swalg_no_match():
    result = smith_waterman(seq1="AAAA", seq2="CCCC", match=2, mismatch=-3, gap=-3)
    assert result.value == 0


def test_cheatsheet():
    from morie.fn.swalg import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
