"""Tests for moirais.fn.zipfl."""
import numpy as np
from moirais.fn.zipfl import zipf_law_fit


def test_zipfl_smoke():
    rng = np.random.default_rng(42)
    result = zipf_law_fit(frequencies=rng.standard_normal(20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.zipfl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
