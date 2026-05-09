"""Tests for moirais.fn.tbund — torus bundle."""

import pytest

from moirais.fn.tbund import torus_bundle


class TestTorusBundle:
    def test_identity(self):
        r = torus_bundle()
        assert r.extra["geometry"] == "Euclidean"

    def test_hyperbolic(self):
        r = torus_bundle([[2, 1], [1, 1]])
        assert r.extra["geometry"] == "Sol"

    def test_nil(self):
        r = torus_bundle([[1, 1], [0, 1]])
        assert r.extra["geometry"] == "Nil"

    def test_bad_det(self):
        with pytest.raises(ValueError):
            torus_bundle([[2, 0], [0, 2]])
