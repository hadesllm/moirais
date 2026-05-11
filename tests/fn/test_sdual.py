"""Test sdual."""
import pytest
from morie.fn.sdual import s_duality


def test_sdual_basic():
    r = s_duality(g_s=0.1)
    assert r.value == pytest.approx(10.0)


def test_sdual_self_dual():
    r = s_duality(g_s=1.0)
    assert r.extra["is_self_dual"]


def test_sdual_invalid():
    with pytest.raises(ValueError):
        s_duality(g_s=0.0)
