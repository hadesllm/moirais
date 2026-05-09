"""Tests for moirais.fn.zssgs -- Sequential Gaussian simulation"""

import numpy as np
import pytest

from moirais.fn.zssgs import seq_gauss_sim


class TestSeqGaussSim:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = seq_gauss_sim(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = seq_gauss_sim(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
