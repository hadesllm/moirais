"""Tests for positive definiteness check."""
import numpy as np
from moirais.fn.sgpdf import sgpdf


def test_sgpdf_positive_definite():
    A = np.array([[2, 1], [1, 2]], dtype=float)
    r = sgpdf(A)
    assert r.name == "positive_definiteness_check"
    assert r.extra["is_positive_definite"] is True


def test_sgpdf_not_positive_definite():
    A = np.array([[1, 2], [2, 1]], dtype=float)
    r = sgpdf(A)
    assert r.extra["is_positive_definite"] is False
    assert r.extra["min_eigenvalue"] < 0
