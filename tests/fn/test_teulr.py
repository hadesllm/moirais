"""Tests for moirais.fn.teulr — Euler characteristic."""

import pytest

from moirais.fn.teulr import torus_euler_char


class TestTorusEulerChar:
    def test_sphere(self):
        assert torus_euler_char(genus=0).value == 2.0

    def test_torus(self):
        assert torus_euler_char(genus=1).value == 0.0

    def test_genus_2(self):
        assert torus_euler_char(genus=2).value == -2.0

    def test_invalid(self):
        with pytest.raises(ValueError):
            torus_euler_char(genus=-1)
