"""Tests for morie.fn.lgrte."""
import numpy as np
from morie.fn.lgrte import lgrte


def test_lgrte_smoke():
    rng = np.random.default_rng(42)
    values = np.abs(rng.standard_normal((5, 10))) + 1.0
    result = lgrte(values=values)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.lgrte import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
