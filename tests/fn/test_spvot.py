"""Tests for morie.fn.spvot — spatial vote."""
import numpy as np
from morie.fn.spvot import spvot


def test_spvot_smoke():
    r = spvot([0.0], [[1.0], [3.0], [5.0]])
    assert r.name == "spatial_vote"
    assert r.value == 0
    assert "distances" in r.extra


def test_spvot_middle():
    r = spvot([2.5], [[0.0], [2.0], [5.0]])
    assert r.value == 1
