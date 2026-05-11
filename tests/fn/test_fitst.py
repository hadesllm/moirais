"""Tests for morie.fn.fitst -- correct classification rate."""
import numpy as np
from morie.fn.fitst import fit_statistic_correct, fitst


def test_alias():
    assert fitst is fit_statistic_correct


def test_perfect():
    r = fit_statistic_correct([1, 0, 1], [1, 0, 1])
    assert r.name == "fit_statistic_correct"
    assert r.value == 1.0


def test_half():
    r = fit_statistic_correct([1, 0, 1, 0], [1, 1, 0, 0])
    assert abs(r.value - 0.5) < 1e-10
