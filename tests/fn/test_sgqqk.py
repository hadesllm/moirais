"""Tests for QQ plot kriging."""
import numpy as np
from moirais.fn.sgqqk import sgqqk


def test_sgqqk_smoke():
    rng = np.random.default_rng(9)
    se = rng.standard_normal(50)
    r = sgqqk(se)
    assert r.name == "qq_plot_kriging"
    assert r.statistic > 0.9
    assert "theoretical" in r.extra
    assert "sample" in r.extra


def test_cheatsheet():
    from moirais.fn.sgqqk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
