"""Tests for moirais.fn.hrmit."""
import numpy as np
from moirais.fn.hrmit import hermite_interp


def test_hrmit_smoke():
    x_known = np.linspace(0, 2*np.pi, 10)
    result = hermite_interp(
        x_known=x_known,
        y_known=np.sin(x_known),
        dy_known=np.cos(x_known),
        x_eval=np.linspace(0.1, 2*np.pi - 0.1, 20)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.hrmit import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
