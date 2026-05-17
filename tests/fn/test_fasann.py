"""We suffer more often in imagination than in reality. — Seneca"""

import numpy as np
from morie.fn.fasann import fast_ann, fasann
from morie.fn._containers import DescriptiveResult


class TestFasann:
    def test_alias(self):
        assert fasann is fast_ann

    def test_basic(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, (100, 5))
        result = fast_ann(data, k=3, seed=42)
        assert isinstance(result, DescriptiveResult)
        assert len(result.value["indices"]) == 3
        assert len(result.value["distances"]) == 3

    def test_first_is_self(self):
        data = np.eye(5)
        result = fast_ann(data, query=data[:1], k=1, seed=42)
        assert result.value["distances"][0] < 0.01
