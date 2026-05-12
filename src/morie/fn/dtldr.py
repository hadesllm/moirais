# morie.fn -- function file (hadesllm/morie)
"""Data loader statistics."""

from __future__ import annotations

from ._containers import DescriptiveResult


def data_loader_stats(
    dataset_size: int,
    batch_size: int,
    seq_len: int,
    n_epochs: int = 1,
) -> DescriptiveResult:
    """Compute training iteration statistics for a data loader.

    :param dataset_size: Total number of tokens in the dataset.
    :param batch_size: Batch size.
    :param seq_len: Sequence length per sample.
    :param n_epochs: Number of training epochs.
    :return: DescriptiveResult with iterations/epoch and total steps.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    tokens_per_batch = batch_size * seq_len
    iters_per_epoch = max(dataset_size // tokens_per_batch, 1)
    total_steps = iters_per_epoch * n_epochs
    total_tokens_seen = total_steps * tokens_per_batch

    return DescriptiveResult(
        name="data_loader_stats",
        value=total_steps,
        extra={
            "iters_per_epoch": iters_per_epoch,
            "total_steps": total_steps,
            "tokens_per_batch": tokens_per_batch,
            "total_tokens_seen": total_tokens_seen,
            "dataset_size": dataset_size,
            "n_epochs": n_epochs,
        },
    )


def cheatsheet() -> str:
    return "data_loader_stats(dataset_size, batch_size, seq_len) -> training iterations"


dtldr = data_loader_stats
