# morie.fn — function file (hadesllm/morie)
"""The man who moves a mountain begins by carrying away small stones. — Confucius"""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def logic_gates(
    inputs: np.ndarray,
    gate_type: str = "AND",
) -> DescriptiveResult:
    """
    Simulate a logic gate over an array of binary inputs.

    Supported gates: AND, OR, NAND, NOR, XOR, XNOR, NOT.
    For NOT, only the first column is used.

    :param inputs: 2D binary array (rows = test vectors, cols = inputs).
    :param gate_type: Gate type string. Default "AND".
    :return: DescriptiveResult with output vector.
    :raises ValueError: If gate_type is unknown or inputs are not binary.

    References
    ----------
    Mano, M. M. (2013). *Digital Design*. 5th ed. Pearson.
    """
    x = np.asarray(inputs, dtype=int)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    unique = np.unique(x)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError("Inputs must be binary (0 or 1).")

    gate = gate_type.upper()
    valid_gates = {"AND", "OR", "NAND", "NOR", "XOR", "XNOR", "NOT"}
    if gate not in valid_gates:
        raise ValueError(f"Unknown gate '{gate_type}'. Choose from {valid_gates}.")

    if gate == "NOT":
        output = 1 - x[:, 0]
    elif gate == "AND":
        output = np.all(x, axis=1).astype(int)
    elif gate == "OR":
        output = np.any(x, axis=1).astype(int)
    elif gate == "NAND":
        output = 1 - np.all(x, axis=1).astype(int)
    elif gate == "NOR":
        output = 1 - np.any(x, axis=1).astype(int)
    elif gate == "XOR":
        output = np.bitwise_xor.reduce(x, axis=1)
    elif gate == "XNOR":
        output = 1 - np.bitwise_xor.reduce(x, axis=1)

    return DescriptiveResult(
        name=f"{gate} Gate",
        value=int(np.sum(output)),
        extra={
            "inputs": x,
            "output": output,
            "gate_type": gate,
            "n_vectors": x.shape[0],
            "n_inputs": x.shape[1],
        },
    )


short = logic_gates


def cheatsheet() -> str:
    return "logic_gates({}) -> Logic gate simulation. 'These aren't the droids you're looki"
