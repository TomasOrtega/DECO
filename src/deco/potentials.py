# src/deco/potentials.py
import math
from abc import ABC, abstractmethod

import numpy as np


class BasePotential(ABC):
    """Abstract base class for a coin-betting potential function."""

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    @abstractmethod
    def F(self, t, x):
        """Evaluates the potential function F_t(x)."""
        pass

    @abstractmethod
    def beta(self, t, x):
        """Calculates the betting fraction beta_t(x)."""
        pass

    def h(self, t, x):
        """Implements the betting function h_t(x) from Definition 3."""
        if t < 1:
            return np.zeros_like(x)

        # Handle vector inputs by taking norm and using normalized direction
        x = np.asarray(x)
        if x.ndim > 0 and x.size > 1:  # Vector case
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-8:  # Zero check
                return np.zeros_like(x)
            direction = x / norm_x
            # Compute h with the scalar norm value
            h_scalar = self.h(t, norm_x)
            return h_scalar * direction
        else:  # Scalar case
            return self.beta(t, x) * self.F(t - 1, x)


class ExponentialPotential(BasePotential):
    """Implements the exponential potential from Table I."""

    def F(self, t, x):
        if t == 0:
            return self.epsilon

        return (self.epsilon / np.sqrt(t)) * np.exp(x**2 / (2 * t))

    def beta(self, t, x):
        return np.tanh(x / t)


class KTPotential(BasePotential):
    """
    Implements the Krichevsky-Trofimov (KT) potential using its
    mathematically simplified form with the Beta function.

    This implementation relies on the standard-library log-gamma function.
    It avoids importing scipy in the hot path and therefore keeps the Docker
    review experiments lightweight while preserving the log-domain computation
    used for numerical stability.
    """

    def __init__(self, epsilon=1.0):
        super().__init__(epsilon)
        # Pre-calculate logs of constants for efficiency.
        self.log_epsilon_over_pi = math.log(epsilon / math.pi)
        self.log_2 = math.log(2.0)

    @staticmethod
    def _betaln(a, b):
        """Compute log(Beta(a,b)) using log-gamma identities."""
        return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

    def F(self, t, x):
        """
        Calculates F_t(x) using the simplified formula:
        F_t(x) = (epsilon * 2^t / pi) * Beta((t+1+|x|)/2, (t+1-|x|)/2)
        """
        # Base case: For t=0, the potential is simply epsilon.
        if t == 0:
            return self.epsilon

        abs_x = abs(float(np.asarray(x)))

        # The domain requires |x| < t + 1.
        if abs_x >= t + 1:
            return float("inf")

        # Define arguments for the Beta function.
        a = (t + 1 + abs_x) / 2
        b = (t + 1 - abs_x) / 2

        # Calculate the log of the potential for max stability.
        # log(F) = log(epsilon/pi) + t*log(2) + log(Beta(a, b))
        log_F = self.log_epsilon_over_pi + t * self.log_2 + self._betaln(a, b)

        # Convert back from log-space to get the final value.
        # Avoid an OverflowError if an experiment leaves the intended domain.
        if log_F > 700:
            return float("inf")
        return math.exp(log_F)

    def beta(self, t, x):
        if t == 0:
            return np.zeros_like(x)
        return x / t
