"""Tests for moirais.fn.txtcl."""
import numpy as np
from moirais.fn.txtcl import text_classify


def test_txtcl_smoke():
    rng = np.random.default_rng(42)
    result = text_classify(
        X_train=rng.standard_normal((30, 3)),
        y_train=rng.integers(0, 2, size=30)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.txtcl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
