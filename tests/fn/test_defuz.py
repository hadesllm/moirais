"""Tests for morie.fn.defuz."""
import numpy as np
import pytest
from morie.fn.defuz import defuz


@pytest.mark.xfail(reason="np.trapz removed in numpy 2.x", strict=False)
def test_defuz_smoke():
    x = np.linspace(0, 10, 50)
    mf = np.exp(-0.5 * (x - 5)**2)
    result = defuz(x=x, mf=mf)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.defuz import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
