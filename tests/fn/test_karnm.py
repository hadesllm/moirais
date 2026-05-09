"""Tests for moirais.fn.karnm."""
import numpy as np
from moirais.fn.karnm import karnaugh_map


def test_karnm_smoke():
    rng = np.random.default_rng(42)
    result = karnaugh_map(truth_table=np.array([1, 0, 1, 0, 1, 1, 0, 1]), n_vars=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.karnm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
