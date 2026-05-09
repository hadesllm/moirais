"""Test sample_entropy (sentr)."""
import numpy as np
from moirais.fn.sentr import sample_entropy, sentr
from moirais.fn._containers import DescriptiveResult


class TestSampleEntropy:
    def test_constant_low(self):
        x = np.ones(50)
        result = sample_entropy(x, m=2, r=0.2)
        assert isinstance(result, DescriptiveResult)

    def test_random_higher(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 50)
        result = sample_entropy(x, m=2, r=0.2)
        assert result.value is not None

    def test_alias(self):
        assert sentr is sample_entropy
