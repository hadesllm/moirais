"""Tests for moirais.fn.fastp."""
import numpy as np
from moirais.fn.fastp import fast_pca


def test_fastp_smoke():
    rng = np.random.default_rng(42)
    result = fast_pca(X=rng.standard_normal((30, 3)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.fastp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
