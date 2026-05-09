"""Tests for moirais.fn.gwpca."""
import numpy as np
from moirais.fn.gwpca import gw_pca


def test_gwpca_smoke():
    rng = np.random.default_rng(42)
    result = gw_pca(X=rng.standard_normal((30, 3)), coords=rng.uniform(size=(30, 2)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.gwpca import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
