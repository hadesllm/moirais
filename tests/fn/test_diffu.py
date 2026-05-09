"""Tests for moirais.fn.diffu."""
import numpy as np
from moirais.fn.diffu import heat_diffusion


def test_diffu_smoke():
    rng = np.random.default_rng(42)
    result = heat_diffusion(T0=np.sin(np.linspace(0, 4*np.pi, 100)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.diffu import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
