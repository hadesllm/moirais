"""Tests for moirais.fn.visnm -- spectral clustering."""

import numpy as np
from moirais.fn.visnm import mind_stone_cluster, visnm
from moirais.fn._containers import DescriptiveResult


class TestVisnm:
    def test_alias(self):
        assert visnm is mind_stone_cluster

    def test_two_clusters(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.3, (20, 2)), rng.normal(3, 0.3, (20, 2))])
        r = mind_stone_cluster(X, k=2, seed=42)
        assert isinstance(r, DescriptiveResult)
        assert len(r.value["labels"]) == 40
        assert len(np.unique(r.value["labels"])) == 2
