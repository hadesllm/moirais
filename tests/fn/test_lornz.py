"""Tests for morie.fn.lornz."""
import numpy as np
import pytest
from morie.fn.lornz import lorenz_curve


@pytest.mark.xfail(reason="np.trapz removed in numpy 2.x", strict=False)
def test_lornz_smoke():
    rng = np.random.default_rng(42)
    incomes = rng.uniform(1000, 100000, size=50)
    result = lorenz_curve(incomes=incomes)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.lornz import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
