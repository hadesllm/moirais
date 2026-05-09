"""Tests for moirais.fn.cltrn -- Cluster trial sample size."""

import pytest
from moirais.fn.cltrn import cluster_trial_size


class TestClusterTrial:
    def test_basic(self):
        res = cluster_trial_size(p1=0.3, p2=0.2, icc=0.05, cluster_size=30)
        assert res.measure == "cluster_trial_size"
        assert res.estimate > 0

    def test_higher_icc_more_clusters(self):
        low = cluster_trial_size(p1=0.3, p2=0.2, icc=0.01, cluster_size=30)
        high = cluster_trial_size(p1=0.3, p2=0.2, icc=0.1, cluster_size=30)
        assert high.estimate >= low.estimate

    def test_invalid(self):
        with pytest.raises(ValueError):
            cluster_trial_size(p1=1.5, p2=0.5, icc=0.05, cluster_size=30)
