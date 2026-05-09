"""Test AdamW optimizer."""
import numpy as np
from moirais.fn.adamw import adamw


def test_adamw_basic():
    """Test basic AdamW step."""
    w = np.array([1.0, 2.0])
    g = np.array([0.1, 0.2])
    result = adamw(w, g)
    assert result["weights"].shape == w.shape
    assert result["t"] == 1


def test_adamw_multiple_steps():
    """Test multiple steps."""
    w = np.array([1.0, 2.0])
    g = np.array([0.1, 0.2])
    result1 = adamw(w, g, t=0)
    result2 = adamw(result1["weights"], g, m=result1["m"], v=result1["v"], t=result1["t"])
    assert result2["t"] == 2
    assert not np.allclose(result1["weights"], result2["weights"])


def test_adamw_weight_decay():
    """Test weight decay."""
    w = np.array([1.0])
    g = np.array([0.0])
    result = adamw(w, g, weight_decay=0.01)
    assert result["weights"][0] < w[0]


def test_adamw_bias_correction():
    """Test bias correction at early timesteps."""
    w = np.array([1.0, 2.0])
    g = np.array([0.1, 0.2])
    result = adamw(w, g, t=0)
    assert result["t"] == 1


def test_adamw_shape_mismatch():
    """Test shape mismatch."""
    w = np.array([1.0, 2.0])
    g = np.array([0.1])
    try:
        adamw(w, g)
        assert False
    except ValueError:
        pass
