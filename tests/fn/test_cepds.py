"""Test cepstral_distance (cepds)."""
import numpy as np
from moirais.fn.cepds import cepstral_distance, cepds
from moirais.fn._containers import DescriptiveResult


class TestCepds:
    def test_identical(self):
        c = [1.0, 0.5, 0.2]
        result = cepstral_distance(c, c)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cepstral_distance"
        assert abs(result.value) < 1e-10

    def test_different(self):
        c1 = [1.0, 0.0]
        c2 = [0.0, 1.0]
        result = cepstral_distance(c1, c2)
        assert abs(result.value - np.sqrt(2)) < 1e-10

    def test_alias(self):
        assert cepds is cepstral_distance
