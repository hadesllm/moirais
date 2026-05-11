"""Tests for morie.fn.lcsub."""
import numpy as np
from morie.fn.lcsub import longest_common_subseq


def test_lcsub_smoke():
    rng = np.random.default_rng(42)
    result = longest_common_subseq(seq1="ACGTACGT", seq2="ACGTACGT")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.lcsub import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
