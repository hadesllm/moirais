"""Tests for recurrence quantification entropy."""
import numpy as np
from moirais.fn.rqent import recurrence_entropy, rqent


def test_sine():
    x = np.sin(np.linspace(0, 8 * np.pi, 300))
    r = recurrence_entropy(x, m=2, eps=0.3)
    assert r.estimate >= 0


def test_alias():
    assert rqent is recurrence_entropy
