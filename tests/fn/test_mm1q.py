"""Tests for morie.fn.mm1q."""
import numpy as np
from morie.fn.mm1q import mm1q


def test_mm1q_smoke():
    result = mm1q(arrival_rate=0.5, service_rate=1.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.mm1q import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
