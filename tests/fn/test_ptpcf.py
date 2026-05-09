"""Tests for moirais.fn.ptpcf -- Pair correlation function g(r)"""

import numpy as np
import pytest

from moirais.fn.ptpcf import pair_corr_fn


class TestPairCorrFn:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pair_corr_fn(data)
        assert result.statistic is not None

    def test_output_type(self):
        result = pair_corr_fn(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "statistic")
