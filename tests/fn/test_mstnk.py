"""Tests for morie.fn.mstnk -- Trustworthiness metric"""

import numpy as np
import pytest

from morie.fn.mstnk import trustworthiness


class TestTrustworthiness:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = trustworthiness(data)
        assert result.value is not None

    def test_output_type(self):
        result = trustworthiness(np.array([1.0, 2.0, 3.0]))
        assert hasattr(result, "value")
