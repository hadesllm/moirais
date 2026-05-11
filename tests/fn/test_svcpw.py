"""Tests for morie.fn.svcpw -- Copeland spatial winner"""

import numpy as np
import pytest

from morie.fn.svcpw import copeland_winner


class TestCopelandWinner:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = copeland_winner(data)
        assert result.value is not None

    def test_output_type(self):
        result = copeland_winner(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
