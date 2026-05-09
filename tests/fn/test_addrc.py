"""Tests for addrc (full adder)."""
import numpy as np
from moirais.fn.addrc import full_adder


def test_full_adder_scalar():
    r = full_adder(1, 1, carry_in=0)
    assert r.extra["carry_out"] == 1


def test_full_adder_array():
    r = full_adder(np.array([1, 0, 1]), np.array([1, 1, 0]))
    assert "sum_bits" in r.extra
