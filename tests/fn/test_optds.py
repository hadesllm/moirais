"""Tests for moirais.fn.optds."""
import numpy as np
from moirais.fn.optds import optds


def test_optds_smoke():
    result = optds(n_points=10, n_factors=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.optds import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
