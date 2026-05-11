"""Test sugrv."""
import pytest
from morie.fn.sugrv import supergravity_action


def test_sugrv_11d():
    r = supergravity_action(d=11, N=1)
    assert r.value == 32.0


def test_sugrv_invalid_dim():
    with pytest.raises(ValueError):
        supergravity_action(d=12)


def test_sugrv_too_many():
    with pytest.raises(ValueError):
        supergravity_action(d=10, N=4)
