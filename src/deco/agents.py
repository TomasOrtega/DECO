# src/deco/agents.py
import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for an online learning agent."""

    def __init__(self, agent_id, dim):
        self.id = agent_id
        self.dim = dim
        self.x = np.zeros(dim)  # Current decision

    @abstractmethod
    def predict(self, t):
        """Predict the next decision x_t."""
        pass

    @abstractmethod
    def update(self, g_t):
        """Perform a local update based on the received gradient g_t."""
        pass

    @abstractmethod
    def gossip(self, network_state, W):
        """Exchange information with neighbors."""
        pass


class DecoAgent(BaseAgent):
    """Implements the proposed Decentralized Coin-betting agent (Algorithm 1)."""

    def __init__(self, agent_id, dim, potential, version="ii"):
        super().__init__(agent_id, dim)
        self.potential = potential
        self.version = version
        self.w = potential.epsilon  # Wealth
        self.G = np.zeros(dim)  # Accumulated (negative) gradients: -sum(g_s)
        self.hat_w, self.hat_G = 0, np.zeros(dim)  # Temp variables for gossip

    def predict(self, t):
        if t == 0:
            return self.x

        if self.version == "i":
            # Version (i): x_t = beta_t(G_{t-1}) * w_{t-1}
            beta_val = self.potential.beta(t, self.G)
            self.x = beta_val * self.w
        else:  # version 'ii'
            # Version (ii): x_t = h_t(G_{t-1})
            self.x = self.potential.h(t, self.G)
        return self.x

    def update(self, g_t):
        c_t = -g_t
        self.hat_w = self.w + np.dot(c_t, self.x)
        self.hat_G = self.G + c_t

    def gossip(self, network_state, W, return_state=False):
        """Gossip the communicated DECO state.

        DECO-i communicates both wealth and accumulated-gradient state. DECO-ii is
        the one-state communication variant: the decision uses h_t(G), so only G
        must be exchanged. This distinction is important for communication-cost
        accounting and for fair DECO-i/DECO-ii comparisons.
        """
        new_state = {}
        if "w" in network_state:
            new_state["w"] = sum(
                W[self.id, j] * network_state["w"][j] for j in range(len(W))
            )
        if "G" in network_state:
            new_state["G"] = sum(
                W[self.id, j] * network_state["G"][j] for j in range(len(W))
            )
        if return_state:
            return new_state
        self.apply_gossip_state(new_state)

    def apply_gossip_state(self, state):
        if "w" in state:
            self.w = state["w"]
        if "G" in state:
            self.G = state["G"]


class DGDAgent(BaseAgent):
    """Baseline: Decentralized Gradient Descent with decreasing learning rate."""

    def __init__(self, agent_id, dim, learning_rate=1.0):
        super().__init__(agent_id, dim)
        self.initial_lr = learning_rate
        self.t = 0  # Time step counter

    def predict(self, t):
        self.t = t
        return self.x

    def update(self, g_t):
        # Use decreasing learning rate: lr_t = initial_lr / sqrt(t)
        # Add 1 to t to avoid division by zero at t=0
        current_lr = self.initial_lr / np.sqrt(self.t + 1)
        self.x = self.x - current_lr * g_t

    def gossip(self, network_state, W, return_state=False):
        # Average decisions with neighbors
        new_x = sum(W[self.id, j] * network_state["x"][j] for j in range(len(W)))
        if return_state:
            return {"x": new_x}
        self.x = new_x

    def apply_gossip_state(self, state):
        self.x = state["x"]


class AdaptiveDGDAgent(BaseAgent):
    """Decentralized adaptive-gradient baseline.

    The reviewer-requested baselines below keep the same decentralized gossip
    interface as DGD while adapting their effective update size from past local
    gradients. They are not parameter-free, but they are useful comparisons for
    testing whether DECO's gain comes from adaptive update magnitudes alone.
    """

    SUPPORTED_METHODS = {"adagrad", "rmsprop", "adam", "adamw", "momentum", "nesterov"}

    def __init__(
        self,
        agent_id,
        dim,
        method="adagrad",
        learning_rate=0.1,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(agent_id, dim)
        self.method = method.lower()
        if self.method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown adaptive method '{method}'. "
                f"Choose from {sorted(self.SUPPORTED_METHODS)}."
            )

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.t = 0
        self.accum_sq = np.zeros(dim)
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.velocity = np.zeros(dim)

    def predict(self, t):
        # Optimizer time steps are one-based for bias correction.
        self.t = t + 1
        return self.x

    def update(self, g_t):
        if self.method == "adagrad":
            self.accum_sq += g_t**2
            step = self.learning_rate * g_t / (np.sqrt(self.accum_sq) + self.epsilon)
            self.x -= step
        elif self.method == "rmsprop":
            self.v = self.beta2 * self.v + (1 - self.beta2) * (g_t**2)
            step = self.learning_rate * g_t / (np.sqrt(self.v) + self.epsilon)
            self.x -= step
        elif self.method in {"adam", "adamw"}:
            self.m = self.beta1 * self.m + (1 - self.beta1) * g_t
            self.v = self.beta2 * self.v + (1 - self.beta2) * (g_t**2)
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)
            if self.method == "adamw" and self.weight_decay > 0:
                self.x *= 1 - self.learning_rate * self.weight_decay
            step = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            self.x -= step
        elif self.method == "momentum":
            self.velocity = self.beta1 * self.velocity + g_t
            self.x -= (self.learning_rate / np.sqrt(self.t)) * self.velocity
        elif self.method == "nesterov":
            prev_velocity = self.velocity.copy()
            self.velocity = self.beta1 * self.velocity + g_t
            lookahead_grad = g_t + self.beta1 * (self.velocity - prev_velocity)
            self.x -= (self.learning_rate / np.sqrt(self.t)) * lookahead_grad

    def gossip(self, network_state, W, return_state=False):
        new_x = sum(W[self.id, j] * network_state["x"][j] for j in range(len(W)))
        if return_state:
            return {"x": new_x}
        self.x = new_x

    def apply_gossip_state(self, state):
        self.x = state["x"]


class CentralizedAgent(DecoAgent):
    """
    Centralized Coin-Betting agent that inherits from DecoAgent.
    Uses DECO-i by default and works with averaged gradients across all agents.
    """

    def __init__(self, agent_id, dim, potential, version="i"):
        super().__init__(agent_id, dim, potential, version)

    def update(self, avg_g_t):
        """Update with averaged gradient across all agents (same as DECO update)"""
        c_t = -avg_g_t
        self.hat_w = self.w + np.dot(c_t, self.x)
        self.hat_G = self.G + c_t
        # Immediately apply the update (no gossip step needed for centralized)
        self.w = self.hat_w
        self.G = self.hat_G

    def gossip(self, *args, **kwargs):
        # No gossip needed for a centralized agent - updates are applied immediately
        pass
