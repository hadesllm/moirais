"""Tests for moirais.fn.tmapn — torus mapping class group."""

import pytest

from moirais.fn.tmapn import torus_mapping_class


class TestTorusMappingClass:
    def test_identity(self):
        r = torus_mapping_class()
        assert r.extra["classification"] == "parabolic"

    def test_hyperbolic(self):
        r = torus_mapping_class([[2, 1], [1, 1]])
        assert r.extra["classification"] == "hyperbolic"
        assert r.extra["trace"] == 3

    def test_elliptic(self):
        r = torus_mapping_class([[0, -1], [1, 0]])
        assert r.extra["classification"] == "elliptic"

    def test_bad_det(self):
        with pytest.raises(ValueError):
            torus_mapping_class([[1, 1], [1, 1]])

    def test_bad_shape(self):
        with pytest.raises(ValueError):
            torus_mapping_class([[1, 0, 0], [0, 1, 0]])
