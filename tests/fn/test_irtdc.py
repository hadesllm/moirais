"""Tests for moirais.fn.irtdc -- IRT discrimination summary."""
import numpy as np
from moirais.fn.irtdc import irt_discrimination_summary, irtdc


def test_alias():
    assert irtdc is irt_discrimination_summary


def test_smoke():
    alpha = [0.5, 1.0, 1.5, 2.0]
    r = irt_discrimination_summary(alpha)
    assert r.name == "irt_discrimination_summary"
    assert abs(r.extra["mean"] - 1.25) < 1e-10
    assert r.extra["n_items"] == 4


def test_single():
    r = irt_discrimination_summary([1.5])
    assert r.extra["min"] == 1.5
    assert r.extra["max"] == 1.5
