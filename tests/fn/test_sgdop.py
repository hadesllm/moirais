"""Test SGD optimizer."""
import numpy as np
from moirais.fn.sgdop import sgdop


def test_sgdop_basic():
    """Test basic SGD step."""
    w = np.array([1.0, 2.0, 3.0])
    g = np.array([0.1, 0.2, 0.3])
    result = sgdop(w, g, learning_rate=0.01, momentum=0.0)
    assert result["weights"].shape == w.shape
    assert not np.allclose(result["weights"], w)


def test_sgdop_momentum():
    """Test SGD with momentum."""
    w = np.array([1.0])
    g = np.array([0.1])
    result = sgdop(w, g, learning_rate=0.01, momentum=0.9)
    assert result["velocity"].shape == w.shape


def test_sgdop_nesterov():
    """Test Nesterov acceleration."""
    w = np.array([1.0, 2.0])
    g = np.array([0.1, 0.2])
    result = sgdop(w, g, learning_rate=0.01, momentum=0.9, nesterov=True)
    assert result["weights"].shape == w.shape


def test_sgdop_momentum_accumulation():
    """Test momentum buffer accumulates."""
    w = np.array([1.0])
    g = np.array([0.1])
    v1 = np.array([0.0])
    result1 = sgdop(w, g, learning_rate=0.01, momentum=0.9, velocity=v1)
    result2 = sgdop(result1["weights"], g, momentum=0.9, velocity=result1["velocity"])
    assert result2["velocity"][0] != result1["velocity"][0]


def test_sgdop_invalid_momentum():
    """Test invalid momentum."""
    w = np.array([1.0])
    g = np.array([0.1])
    try:
        sgdop(w, g, momentum=1.5)
        assert False
    except ValueError:
        pass


def test_sgdop_shape_mismatch():
    """Test shape mismatch."""
    w = np.array([1.0, 2.0])
    g = np.array([0.1])
    try:
        sgdop(w, g)
        assert False
    except ValueError:
        pass
