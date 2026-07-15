"""Shared configurations for the online decentralized baseline experiments."""

from copy import deepcopy

import numpy as np


ADAPTIVE_BASELINE_NAMES = ("D-AdaGrad", "D-RMSProp", "D-Adam")
ONLINE_BASELINE_NAMES = ("DOGD", *ADAPTIVE_BASELINE_NAMES)
LEARNING_RATES = np.logspace(-3, 3, num=25)
REFERENCE_LEARNING_RATE = 0.1

_BASELINE_CONFIGS = {
    "DOGD": {"agent_type": "DGD"},
    "D-AdaGrad": {"agent_type": "AdaptiveDGD", "method": "adagrad"},
    "D-RMSProp": {
        "agent_type": "AdaptiveDGD",
        "method": "rmsprop",
        "beta2": 0.99,
    },
    "D-Adam": {"agent_type": "AdaptiveDGD", "method": "adam"},
}


def make_online_baseline_config(name, learning_rate, **overrides):
    """Return one causal online baseline configuration at a base rate."""
    try:
        config = deepcopy(_BASELINE_CONFIGS[name])
    except KeyError as error:
        raise ValueError(f"Unknown online baseline: {name}") from error
    config["lr"] = float(learning_rate)
    config.update(overrides)
    return config


def cumulative_network_loss(history):
    return float(np.sum(history["network_loss"]))


def best_rate_and_loss(rate_results):
    """Select the finite minimum from a displayed learning-rate sweep."""
    candidates = (
        (float(rate), cumulative_network_loss(history))
        for rate, history in rate_results.items()
    )
    finite = [(rate, loss) for rate, loss in candidates if np.isfinite(loss)]
    if not finite:
        raise ValueError("The learning-rate sweep contains no finite losses")
    return min(finite, key=lambda item: item[1])
