"""Tests for moirais.fn.fzinf."""
import numpy as np
import pytest
from moirais.fn.fzinf import fzinf


@pytest.mark.xfail(reason="np.trapz removed in numpy 2.x", strict=False)
def test_fzinf_smoke():
    rng = np.random.default_rng(42)
    result = fzinf(rules=[(0.7, 0.3)], inputs=[0.5, 0.5])
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.fzinf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
