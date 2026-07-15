from collections.abc import Iterable, Mapping

import numpy as np


def mask_large_losses(
    losses_by_algorithm: Mapping[str, Iterable[float]], threshold_factor: float = 5.0
) -> dict[str, np.ndarray]:
    """Replace losses above a shared relative threshold with NaN."""
    if threshold_factor <= 0:
        raise ValueError("threshold_factor must be positive")

    masked = {
        name: np.asarray(losses, dtype=float).copy()
        for name, losses in losses_by_algorithm.items()
    }
    finite_losses = [values[np.isfinite(values)] for values in masked.values()]
    finite_losses = [values for values in finite_losses if values.size]
    if not finite_losses:
        return masked

    threshold = threshold_factor * min(np.min(values) for values in finite_losses)
    for values in masked.values():
        values[(values > threshold) | ~np.isfinite(values)] = np.nan
    return masked
