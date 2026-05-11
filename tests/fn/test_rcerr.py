"""Tests for morie.fn.rcerr -- roll-call errors."""
import numpy as np
from morie.fn.rcerr import roll_call_errors, rcerr


def test_alias():
    assert rcerr is roll_call_errors


def test_smoke():
    pred = np.array([1, 0, 1, 0])
    obs = np.array([1, 0, 0, 1])
    r = roll_call_errors(pred, obs)
    assert r.name == "roll_call_errors"
    assert r.value == 2.0


def test_type_errors():
    pred = np.array([1, 0])
    obs = np.array([0, 1])
    r = roll_call_errors(pred, obs)
    assert r.extra["type_i"] == 1
    assert r.extra["type_ii"] == 1
