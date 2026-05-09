"""Tests for moirais.fn.gates."""
import numpy as np
from moirais.fn.gates import logic_gates


def test_gates_smoke():
    rng = np.random.default_rng(42)
    result = logic_gates(inputs=np.array([[0,0],[0,1],[1,0],[1,1]]))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.gates import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
