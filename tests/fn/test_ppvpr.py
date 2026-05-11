"""Tests for morie.fn.ppvpr."""
import numpy as np
from morie.fn.ppvpr import ppvpr


def test_ppvpr_smoke():
    rng = np.random.default_rng(42)
    result = ppvpr(sensitivity=0.9, specificity=0.8, prevalence=0.1)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.ppvpr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
