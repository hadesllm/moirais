"""Tests for morie.fn.nmwnl -- W-NOMINATE log-likelihood"""

import numpy as np
import pytest

from morie.fn.nmwnl import wnominate_loglik


class TestWnominateLoglik:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = wnominate_loglik(data)
        assert result.value is not None

    def test_output_type(self):
        result = wnominate_loglik(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
