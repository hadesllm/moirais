"""Tests for bootstrap_two_sample."""
import numpy as np, pytest
from morie.fn.boot2 import bootstrap_two_sample


class TestBootstrapTwoSample:
    def test_different_means(self):
        r = bootstrap_two_sample([1, 2, 3, 4, 5], [20, 21, 22, 23, 24], n_boot=999)
        assert r.test_name == "Two-sample bootstrap test"
        assert r.p_value < 0.05

    def test_same_means(self):
        r = bootstrap_two_sample([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], n_boot=999)
        assert r.p_value > 0.1

    def test_empty(self):
        with pytest.raises(ValueError):
            bootstrap_two_sample([], [1, 2])
