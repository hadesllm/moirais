"""Tests for morie.fn.eladef -- elastic deformation field."""

from morie.fn.eladef import elastic_deformation, eladef
from morie.fn._containers import DescriptiveResult


class TestEladef:
    def test_alias(self):
        assert eladef is elastic_deformation

    def test_shape(self):
        r = elastic_deformation((32, 32), alpha=10, sigma=3, seed=42)
        assert isinstance(r, DescriptiveResult)
        assert r.value["dx"].shape == (32, 32)
        assert r.value["dy"].shape == (32, 32)

    def test_magnitude(self):
        r = elastic_deformation((20, 20), alpha=5, seed=0)
        assert r.value["max_displacement"] > 0
        assert r.value["mean_displacement"] > 0
