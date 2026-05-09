"""Tests for moirais.fn.tranm -- transmission tree reconstruction."""

import numpy as np
import pytest
from moirais.fn.tranm import transmission_tree


class TestTransmissionTree:
    def test_chain(self):
        times = np.array([0.0, 3.0, 6.0, 9.0])
        si = np.array([0.0, 0.1, 0.3, 0.6])
        res = transmission_tree(times, si)
        assert res["n_cases"] == 4
        assert res["infector"][0] == -1 or res["infector"][0] >= 0

    def test_ri_sum(self):
        times = np.array([0.0, 2.0, 4.0, 5.0])
        si = np.array([0.0, 0.3, 0.5, 0.2])
        res = transmission_tree(times, si)
        Ri = res["Ri"]
        P = res["probability_matrix"]
        np.testing.assert_allclose(Ri, np.sum(P, axis=0), atol=1e-10)

    def test_probability_rows_sum_to_one(self):
        times = np.array([0.0, 3.0, 5.0])
        si = np.array([0.0, 0.2, 0.5, 0.2, 0.1])
        res = transmission_tree(times, si)
        P = res["probability_matrix"]
        for i in range(len(times)):
            row_sum = np.sum(P[i])
            if row_sum > 0:
                assert row_sum == pytest.approx(1.0, abs=1e-10)

    def test_single_case_raises(self):
        with pytest.raises(ValueError):
            transmission_tree(np.array([0.0]), np.array([0.5, 0.5]))

    def test_ids_passed(self):
        times = np.array([0.0, 2.0, 4.0])
        si = np.array([0.0, 0.5, 0.5])
        res = transmission_tree(times, si, ids=["A", "B", "C"])
        assert res["ids"] == ["A", "B", "C"]

    def test_infector_index_range(self):
        times = np.array([0.0, 1.0, 3.0, 5.0, 7.0])
        si = np.array([0.0, 0.3, 0.5, 0.2])
        res = transmission_tree(times, si)
        for inf in res["infector"]:
            assert inf == -1 or (0 <= inf < 5)
