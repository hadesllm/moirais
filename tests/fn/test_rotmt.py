"""Tests for moirais.fn.rotmt -- rotation matrix 2D."""

import numpy as np
from moirais.fn.rotmt import rotation_matrix_2d, rotmt


def test_rotmt_identity():
    r = rotmt(0.0)
    assert r.name == "rotation_matrix_2d"
    assert np.allclose(r.value, np.eye(2))


def test_rotmt_90deg():
    r = rotmt(np.pi / 2)
    expected = np.array([[0, -1], [1, 0]], dtype=float)
    assert np.allclose(r.value, expected, atol=1e-10)


def test_rotmt_alias():
    assert rotmt is rotation_matrix_2d
