"""Tests for integration (neural complexity)."""
import numpy as np
import pytest
from moirais.fn.intgn import integration, intgn


def test_diagonal():
    C = np.diag([1.0, 2.0, 3.0])
    r = integration(C)
    assert r.estimate == pytest.approx(0.0, abs=1e-10)


def test_correlated():
    C = np.array([[1.0, 0.5], [0.5, 1.0]])
    r = integration(C)
    assert r.estimate > 0


def test_alias():
    assert intgn is integration


def test_not_square():
    with pytest.raises(ValueError):
        integration(np.ones((2, 3)))
