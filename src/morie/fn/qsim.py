# morie.fn -- function file (hadesllm/morie)
"""Discrete-event M/M/1 queue simulation."""

from __future__ import annotations

import numpy as np

from ._containers import DescriptiveResult


def queue_simulate(
    arrival_rate: float,
    service_rate: float,
    n_customers: int = 1000,
    seed: int = 42,
) -> DescriptiveResult:
    """Discrete-event simulation of an M/M/1 queue.

    Generates exponential inter-arrival and service times, then
    computes empirical waiting statistics.

    Parameters
    ----------
    arrival_rate : float
        Mean arrival rate (lambda > 0).
    service_rate : float
        Mean service rate (mu > 0).
    n_customers : int, default 1000
        Number of customers to simulate.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    DescriptiveResult
        ``value`` is the empirical mean waiting time in queue.
        ``extra`` has ``mean_system_time``, ``mean_queue_length``,
        ``utilization``, ``wait_times`` (array), ``system_times``
        (array).

    Raises
    ------
    ValueError
        If rates are non-positive or n_customers < 1.

    References
    ----------
    Banks, J., Carson, J. S., Nelson, B. L., & Nicol, D. M. (2010).
    *Discrete-Event System Simulation* (5th ed.). Prentice Hall.
    """
    if arrival_rate <= 0:
        raise ValueError(f"arrival_rate must be > 0, got {arrival_rate}.")
    if service_rate <= 0:
        raise ValueError(f"service_rate must be > 0, got {service_rate}.")
    if n_customers < 1:
        raise ValueError(f"n_customers must be >= 1, got {n_customers}.")

    rng = np.random.default_rng(seed)
    inter_arrivals = rng.exponential(1.0 / arrival_rate, n_customers)
    service_times = rng.exponential(1.0 / service_rate, n_customers)

    arrival_times = np.cumsum(inter_arrivals)
    start_times = np.zeros(n_customers)
    depart_times = np.zeros(n_customers)

    start_times[0] = arrival_times[0]
    depart_times[0] = start_times[0] + service_times[0]

    for i in range(1, n_customers):
        start_times[i] = max(arrival_times[i], depart_times[i - 1])
        depart_times[i] = start_times[i] + service_times[i]

    wait_times = start_times - arrival_times
    system_times = depart_times - arrival_times
    total_time = depart_times[-1] - arrival_times[0]
    busy_time = service_times.sum()

    return DescriptiveResult(
        name="QueueSimulation",
        value=float(np.mean(wait_times)),
        extra={
            "mean_system_time": float(np.mean(system_times)),
            "mean_queue_length": float(np.mean(wait_times) * arrival_rate),
            "utilization": float(busy_time / total_time) if total_time > 0 else 0.0,
            "n_customers": n_customers,
            "wait_times": wait_times,
            "system_times": system_times,
        },
    )


qsim = queue_simulate


def cheatsheet() -> str:
    return "queue_simulate({}) -> Discrete-event M/M/1 queue simulation."
