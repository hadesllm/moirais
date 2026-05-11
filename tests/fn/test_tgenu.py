"""Tests for morie.fn.tgenu — genus computation."""

import pytest

from morie.fn.tgenu import torus_genus


class TestTorusGenus:
    def test_from_handles(self):
        assert torus_genus(handles=1).value == 1.0

    def test_from_euler(self):
        r = torus_genus(vertices=4, edges=6, faces=4)
        assert r.value == 0.0

    def test_torus_triangulation(self):
        r = torus_genus(vertices=1, edges=3, faces=2)
        assert r.value == 1.0

    def test_invalid_handles(self):
        with pytest.raises(ValueError):
            torus_genus(handles=-1)

    def test_no_args(self):
        with pytest.raises(ValueError):
            torus_genus()
