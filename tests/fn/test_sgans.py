"""Tests for anisotropy correction."""
import numpy as np
from morie.fn.sgans import sgans


def test_sgans_smoke():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    r = sgans(coords, ratio=2.0, angle=45.0)
    assert r.name == "anisotropy_correction"
    assert "corrected_coords" in r.extra
    assert r.extra["corrected_coords"].shape == (4, 2)


def test_sgans_identity():
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=float)
    r = sgans(coords, ratio=1.0, angle=0.0)
    assert np.allclose(r.extra["corrected_coords"], coords, atol=1e-10)
