"""Tests for moirais.fn.zssis -- Sequential indicator simulation"""

import numpy as np
import pytest

from moirais.fn.zssis import seq_ind_sim


class TestSeqIndSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = seq_ind_sim(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = seq_ind_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
