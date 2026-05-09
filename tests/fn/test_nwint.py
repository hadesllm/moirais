"""Tests for moirais.fn.nwint."""
import numpy as np
from moirais.fn.nwint import newton_interp


def test_nwint_smoke():
    x_known = np.linspace(0, 2*np.pi, 10)
    result = newton_interp(
        x_known=x_known,
        y_known=np.sin(x_known),
        x_eval=np.linspace(0.1, 2*np.pi - 0.1, 20)
    )
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.nwint import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
