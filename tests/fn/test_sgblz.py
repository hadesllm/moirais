"""Tests for Boltzmann acceptance."""
import numpy as np
from moirais.fn.sgblz import sgblz


def test_sgblz_accept_lower():
    r = sgblz(10.0, 5.0, 1.0)
    assert r.name == "boltzmann_accept"
    assert r.statistic == 1.0


def test_sgblz_reject_higher():
    r = sgblz(5.0, 100.0, 0.01)
    assert r.statistic < 0.01


def test_sgblz_zero_temp():
    r = sgblz(5.0, 10.0, 0.0)
    assert r.statistic == 0.0
