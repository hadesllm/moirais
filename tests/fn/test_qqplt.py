"""Tests for moirais.fn.qqplt."""
import numpy as np
from moirais.fn.qqplt import qq_data


def test_qqplt_smoke():
    rng = np.random.default_rng(42)
    result = qq_data(x=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.qqplt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
