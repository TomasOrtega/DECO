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
        # Gossip wealth and accumulated gradients
        new_w = sum(W[self.id, j] * network_state["w"][j] for j in range(len(W)))
        new_G = sum(W[self.id, j] * network_state["G"][j] for j in range(len(W)))
        if return_state:
            return {"w": new_w, "G": new_G}
        self.w = new_w
        self.G = new_G

    def apply_gossip_state(self, state):
        self.w = state["w"]
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
