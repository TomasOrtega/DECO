# src/deco/algorithms.py
import numpy as np
from tqdm import tqdm

from .agents import AdaptiveDGDAgent, CentralizedAgent, DecoAgent, DGDAgent
from .metrics import calculate_losses


def _off_diagonal_messages(W):
    """Count directed neighbor-to-neighbor scalar-message recipients per gossip round."""
    W_without_self = np.asarray(W) - np.diag(np.diag(W))
    return int(np.count_nonzero(W_without_self))


def _state_dimension(config, DIM):
    """Number of real scalars communicated by one directed message."""
    agent_type = config["agent_type"]
    if agent_type == "Deco":
        # DECO-i sends (w, G); DECO-ii sends only G.
        return DIM + 1 if config.get("version") == "i" else DIM
    if agent_type in {"DGD", "AdaptiveDGD"}:
        return DIM
    return 0


def run_simulation(T, N, DIM, env, W, config, u_star):
    """Runs a full online learning simulation."""

    # Initialize agents based on config
    agents = []
    potential = config.get("potential")
    for i in range(N):
        if config["agent_type"] == "Deco":
            agents.append(DecoAgent(i, DIM, potential, config["version"]))
        elif config["agent_type"] == "DGD":
            agents.append(DGDAgent(i, DIM, config["lr"]))
        elif config["agent_type"] == "AdaptiveDGD":
            agents.append(
                AdaptiveDGDAgent(
                    i,
                    DIM,
                    method=config.get("method", "adagrad"),
                    learning_rate=config.get("lr", 0.1),
                    beta1=config.get("beta1", 0.9),
                    beta2=config.get("beta2", 0.999),
                    epsilon=config.get("epsilon", 1e-8),
                    weight_decay=config.get("weight_decay", 0.0),
                )
            )
        elif config["agent_type"] == "Centralized":
            # All "agents" are clones of the same centralized logic
            agents.append(CentralizedAgent(i, DIM, potential))
        else:
            raise ValueError(f"Unknown agent_type: {config['agent_type']}")

    history_dtype = [
        ("local_loss", "f8"),
        ("network_loss", "f8"),
        ("communication_scalars", "f8"),
    ]
    history = np.zeros(T, dtype=history_dtype)

    is_centralized = config["agent_type"] == "Centralized"
    use_gossip = config.get("gossip", False)

    # Get the gossip schedule, q(t). Default to a constant 1.
    gossip_schedule = config.get("q_t", lambda t: 1)
    per_gossip_scalars = _off_diagonal_messages(W) * _state_dimension(config, DIM)
    gossip_powers = {}

    for t in tqdm(range(T), leave=False, disable=config.get("disable_tqdm", False)):
        decisions = [agent.predict(t) for agent in agents]

        # In centralized setting, all agents make the same prediction
        if is_centralized:
            decisions = [decisions[0]] * N

        losses, grads, features, labels = env.get_context(t, decisions)

        # Store the average local loss and the network loss
        avg_local_loss, avg_network_loss = calculate_losses(
            losses, decisions, features, labels
        )
        history["local_loss"][t] = avg_local_loss
        history["network_loss"][t] = avg_network_loss

        # Agent updates
        if is_centralized:
            avg_grad = np.mean(grads, axis=0)
            agents[0].update(avg_grad)
        else:
            for i in range(N):
                agents[i].update(grads[i])

        # Gossip step
        q = 0
        if use_gossip and N > 1:
            # Determine the number of gossip rounds for this timestep
            q = max(int(gossip_schedule(t)), 0)
            history["communication_scalars"][t] = q * per_gossip_scalars

        if q > 0:
            # W**q is exactly q synchronous linear gossip rounds.
            if q not in gossip_powers:
                gossip_powers[q] = np.linalg.matrix_power(W, q)
            mixing = gossip_powers[q]
            if config["agent_type"] == "Deco":
                gradients = mixing @ np.asarray([agent.hat_G for agent in agents])
                if config.get("version") == "i":
                    wealth = mixing @ np.asarray([agent.hat_w for agent in agents])
                for i, agent in enumerate(agents):
                    state = {"G": gradients[i]}
                    if config.get("version") == "i":
                        state["w"] = wealth[i]
                    agent.apply_gossip_state(state)
            elif config["agent_type"] in {"DGD", "AdaptiveDGD"}:
                decisions = mixing @ np.asarray([agent.x for agent in agents])
                for i, agent in enumerate(agents):
                    agent.apply_gossip_state({"x": decisions[i]})
        elif config["agent_type"] == "Deco":
            for agent in agents:
                state = {"G": agent.hat_G}
                if config.get("version") == "i":
                    state["w"] = agent.hat_w
                agent.apply_gossip_state(state)

    return history
