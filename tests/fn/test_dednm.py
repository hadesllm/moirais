"""Test dednm."""
import pytest
from moirais.fn.dednm import dedekind_sum


def test_dednm_basic():
    r = dedekind_sum(h=1, k=7)
    assert r.value is not None
    assert r.name == "dedekind_sum"


def test_dednm_invalid():
    with pytest.raises(ValueError):
        dedekind_sum(h=1, k=0)
