"""Tests for moirais.fn.ripk."""
import numpy as np
from moirais.fn.ripk import ripley_k_corrected


def test_ripk_smoke():
    rng = np.random.default_rng(42)
    result = ripley_k_corrected(points=rng.uniform(size=(20, 2)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.ripk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
