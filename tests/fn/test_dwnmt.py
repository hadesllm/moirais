"""Test dwnmt."""
import numpy as np
from morie.fn.dwnmt import dw_nominate_score


def test_dwnmt_basic():
    r = dw_nominate_score()
    assert r.value is not None


def test_dwnmt_name():
    r = dw_nominate_score()
    assert r.name
