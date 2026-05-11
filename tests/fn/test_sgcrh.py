"""Test sgcrh."""
import numpy as np
from morie.fn.sgcrh import cressie_hawkins


def test_sgcrh_basic():
    r = cressie_hawkins()
    assert r.statistic is not None


def test_sgcrh_name():
    r = cressie_hawkins()
    assert r.name
