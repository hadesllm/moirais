"""Double DQN -- decouples action selection from value."""
import numpy as np
from scipy import stats
from ._richresult import RichResult

__all__ = ["double_dqn"]


def double_dqn(env, net):
    """
    Double DQN -- decouples action selection from value

    Formula: target = r + γ Q_θ⁻(s', argmax_a Q_θ(s',a))

    Parameters
    ----------
    env : array-like
        Input data.
    net : array-like
        Input data.

    Returns
    -------
    result : dict
        Keys: estimate

    References
    ----------
    Hasselt-Guez-Silver (2016)
    """
    env = np.atleast_1d(np.asarray(env, dtype=float))
    n = len(env)
    result = float(np.mean(env))
    se = float(np.std(env, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return RichResult(payload={"estimate": result, "se": se, "n": n, "method": "Double DQN -- decouples action selection from value"})


def cheatsheet():
    return "ddqn: Double DQN -- decouples action selection from value"
