"""Biomedical signal modeling backends from Rangayyan & Krishnan Ch. 7.

Implements: SMUAP point process, ARMA estimation (Newton-Raphson, modified Y-W,
Steiglitz-McBride), AR formant extraction, vocal tract model, Hodgkin-Huxley neuron,
bidomain cardiac model.
"""

from __future__ import annotations

import numpy as np


def smuap_point_process(
    n_mus: int = 10,
    firing_rates: np.ndarray | None = None,
    fs: float = 1000.0,
    duration: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng()
    n_samples = int(fs * duration)
    emg = np.zeros(n_samples)
    if firing_rates is None:
        firing_rates = rng.uniform(5, 30, n_mus)
    events = np.zeros(n_samples)
    muap_len = int(0.01 * fs)
    t_muap = np.arange(muap_len) / fs
    for i in range(n_mus):
        rate = firing_rates[i]
        amplitude = rng.uniform(0.5, 2.0)
        n_firings = int(rate * duration)
        spike_times = np.sort(
            rng.choice(
                n_samples - muap_len,
                size=min(n_firings, n_samples // 2),
                replace=False,
            )
        )
        muap = amplitude * np.sin(2 * np.pi * 500 * t_muap) * np.exp(-300 * t_muap)
        for st in spike_times:
            emg[st : st + muap_len] += muap
            events[st] += 1
    return emg, events


def arma_newton_raphson(
    x: np.ndarray,
    p: int = 4,
    q: int = 2,
    n_iter: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(x)
    a = np.zeros(p)
    b = np.zeros(q + 1)
    b[0] = 1.0
    for _ in range(n_iter):
        e = np.zeros(n)
        for i in range(max(p, q + 1), n):
            ar_part = np.dot(a, x[i - p : i][::-1]) if p > 0 else 0
            ma_part = np.dot(b[1:], e[i - q : i][::-1]) if q > 0 else 0
            e[i] = x[i] - ar_part - ma_part
        for j in range(p):
            grad = 0.0
            for i in range(max(p, q + 1), n):
                grad += e[i] * x[i - j - 1]
            a[j] += 0.001 * grad
        for j in range(1, q + 1):
            grad = 0.0
            for i in range(max(p, q + 1), n):
                grad += e[i] * e[i - j]
            b[j] += 0.001 * grad
    sigma2 = float(np.mean(e[max(p, q + 1) :] ** 2))
    return a, b, sigma2


def modified_yule_walker_arma(
    x: np.ndarray,
    p: int = 4,
    q: int = 2,
) -> tuple[np.ndarray, float]:
    n = len(x)
    r = np.correlate(x, x, mode="full")
    r = r[n - 1 :] / n
    R = np.zeros((p, p))
    rhs = np.zeros(p)
    for i in range(p):
        rhs[i] = r[q + 1 + i]
        for j in range(p):
            R[i, j] = r[abs(q + 1 + i - j - 1)]
    try:
        a = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        a = np.linalg.lstsq(R, rhs, rcond=None)[0]
    sigma2 = float(r[0] - np.dot(a, r[1 : p + 1]))
    return a, max(sigma2, 1e-10)


def steiglitz_mcbride(
    x: np.ndarray,
    p: int = 4,
    q: int = 2,
    n_iter: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    n = len(x)
    a = np.zeros(p)
    b = np.zeros(q + 1)
    b[0] = 1.0
    for _ in range(n_iter):
        filtered_x = x.copy()
        for i in range(p, n):
            filtered_x[i] = x[i] + np.dot(a, filtered_x[i - p : i][::-1])
        r = np.correlate(filtered_x, filtered_x, mode="full")
        mid = len(r) // 2
        r = r[mid:] / n
        R_mat = np.zeros((p, p))
        rhs = np.zeros(p)
        for i in range(p):
            rhs[i] = r[i + 1]
            for j in range(p):
                R_mat[i, j] = r[abs(i - j)]
        try:
            a = np.linalg.solve(R_mat, rhs)
        except np.linalg.LinAlgError:
            break
    sigma2 = float(r[0] - np.dot(a, r[1 : p + 1])) if p > 0 else float(r[0])
    return a, b, max(sigma2, 1e-10)


def ar_formant_extraction(
    x: np.ndarray,
    fs: float = 8000.0,
    order: int = 12,
) -> np.ndarray:
    n = len(x)
    r = np.correlate(x, x, mode="full")[n - 1 : n + order] / n
    R = np.zeros((order, order))
    rhs = np.zeros(order)
    for i in range(order):
        rhs[i] = r[i + 1]
        for j in range(order):
            R[i, j] = r[abs(i - j)]
    try:
        a = np.linalg.solve(R, rhs)
    except np.linalg.LinAlgError:
        return np.array([])
    roots = np.roots(np.concatenate(([1], -a)))
    roots = roots[np.imag(roots) > 0]
    angles = np.angle(roots)
    formants = np.sort(angles * fs / (2 * np.pi))
    return formants[formants > 0]


def vocal_tract_model(
    area_function: np.ndarray,
    fs: float = 8000.0,
    n_samples: int = 1024,
) -> np.ndarray:
    n_sections = len(area_function)
    reflection = np.zeros(n_sections - 1)
    for i in range(n_sections - 1):
        s = area_function[i] + area_function[i + 1]
        if s == 0:
            reflection[i] = 0
        else:
            reflection[i] = (area_function[i + 1] - area_function[i]) / s
    impulse = np.zeros(n_samples)
    impulse[0] = 1.0
    output = impulse.copy()
    for i in range(1, min(n_sections - 1, n_samples)):
        for j in range(n_sections - 2):
            output[i] += reflection[j] * output[max(0, i - j - 1)]
    return output


def hodgkin_huxley(
    duration: float = 50.0,
    dt: float = 0.01,
    I_ext: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    C_m = 1.0
    g_Na, g_K, g_L = 120.0, 36.0, 0.3
    E_Na, E_K, E_L = 50.0, -77.0, -54.4
    steps = int(duration / dt)
    V = np.zeros(steps)
    V[0] = -65.0
    t = np.arange(steps) * dt

    def _alpha_m(v):
        return 0.1 * (v + 40) / (1 - np.exp(-(v + 40) / 10)) if abs(v + 40) > 1e-7 else 1.0

    def _beta_m(v):
        return 4.0 * np.exp(-(v + 65) / 18)

    def _alpha_h(v):
        return 0.07 * np.exp(-(v + 65) / 20)

    def _beta_h(v):
        return 1.0 / (1 + np.exp(-(v + 35) / 10))

    def _alpha_n(v):
        return 0.01 * (v + 55) / (1 - np.exp(-(v + 55) / 10)) if abs(v + 55) > 1e-7 else 0.1

    def _beta_n(v):
        return 0.125 * np.exp(-(v + 65) / 80)

    m = _alpha_m(V[0]) / (_alpha_m(V[0]) + _beta_m(V[0]))
    h = _alpha_h(V[0]) / (_alpha_h(V[0]) + _beta_h(V[0]))
    n = _alpha_n(V[0]) / (_alpha_n(V[0]) + _beta_n(V[0]))

    for i in range(1, steps):
        v = V[i - 1]
        I_Na = g_Na * m**3 * h * (v - E_Na)
        I_K = g_K * n**4 * (v - E_K)
        I_L = g_L * (v - E_L)
        dV = (I_ext - I_Na - I_K - I_L) / C_m
        V[i] = v + dt * dV
        m += dt * (_alpha_m(v) * (1 - m) - _beta_m(v) * m)
        h += dt * (_alpha_h(v) * (1 - h) - _beta_h(v) * h)
        n += dt * (_alpha_n(v) * (1 - n) - _beta_n(v) * n)
        m = np.clip(m, 0, 1)
        h = np.clip(h, 0, 1)
        n = np.clip(n, 0, 1)
    return t, V


def bidomain_model(
    n_cells: int = 50,
    duration: float = 10.0,
    dt: float = 0.1,
    conductivity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    steps = int(duration / dt)
    V = np.zeros((steps, n_cells))
    V[0, :] = -85.0
    V[0, 0:3] = 20.0
    t = np.arange(steps) * dt
    for i in range(1, steps):
        for j in range(n_cells):
            v = V[i - 1, j]
            I_ion = -0.5 * v * (1 - v / 50) * (1 + v / 85)
            diffusion = 0.0
            if j > 0:
                diffusion += conductivity * (V[i - 1, j - 1] - v)
            if j < n_cells - 1:
                diffusion += conductivity * (V[i - 1, j + 1] - v)
            V[i, j] = v + dt * (I_ion + diffusion)
    return t, V
