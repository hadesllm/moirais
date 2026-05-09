"""Tests for permutation_test_two."""
import numpy as np, pytest
from moirais.fn.prmts import permutation_test_two


class TestPermutationTwo:
    def test_different_groups(self):
        r = permutation_test_two([1, 2, 3, 4, 5], [20, 21, 22, 23, 24], n_perm=999)
        assert r.test_name == "Permutation test (two-sample)"
        assert r.p_value < 0.05

    def test_same_groups(self):
        r = permutation_test_two([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], n_perm=999)
        assert r.p_value > 0.1

    def test_empty(self):
        with pytest.raises(ValueError):
            permutation_test_two([], [1])
