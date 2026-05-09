"""Tests for moirais.fn.nwalg."""
import numpy as np
from moirais.fn.nwalg import needleman_wunsch


def test_nwalg_smoke():
    rng = np.random.default_rng(42)
    result = needleman_wunsch(seq1="ACGTACGT", seq2="ACGTACGT")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.nwalg import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
