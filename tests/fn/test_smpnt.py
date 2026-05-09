"""Tests for sample entropy."""
import numpy as np
import pytest
from moirais.fn.smpnt import sample_entropy, smpnt


def test_regular():
    x = np.sin(np.linspace(0, 4 * np.pi, 200))
    r = sample_entropy(x, m=2)
    assert np.isfinite(r.estimate) or r.estimate == float("inf")


def test_alias():
    assert smpnt is sample_entropy


def test_too_short():
    with pytest.raises(ValueError):
        sample_entropy([1, 2])
