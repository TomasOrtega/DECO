# src/deco/algorithms.py
import numpy as np
from tqdm import tqdm
from .metrics import calculate_losses
from .agents import DecoAgent, DGDAgent, CentralizedAgent


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
        elif config["agent_type"] == "Centralized":
            # All "agents" are clones of the same centralized logic
            agents.append(CentralizedAgent(i, DIM, potential))

    history_dtype = [("local_loss", "f8"), ("network_loss", "f8")]
    history = np.zeros(T, dtype=history_dtype)

    is_centralized = config["agent_type"] == "Centralized"
    use_gossip = config.get("gossip", False)

    # Get the gossip schedule, q(t). Default to a constant 1.
    gossip_schedule = config.get("q_t", lambda t: 1)

    for t in tqdm(range(T), leave=False):
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
        if use_gossip and N > 1:
            # Determine the number of gossip rounds for this timestep
            q = gossip_schedule(t)

            # Apply gossip q times
            for _ in range(q):
                # Gather state from all agents before gossiping
                network_state = {}
                if config["agent_type"] == "Deco":
                    # For DECO, we gossip the intermediate values hat_w and hat_G
                    # After the first round, we gossip the newly updated w and G
                    if _ == 0:
                        network_state["w"] = [agent.hat_w for agent in agents]
                        network_state["G"] = [agent.hat_G for agent in agents]
                    else:
                        network_state["w"] = [agent.w for agent in agents]
                        network_state["G"] = [agent.G for agent in agents]

                elif config["agent_type"] == "DGD":
                    network_state["x"] = [agent.x for agent in agents]

                # Create a temporary list to store post-gossip states
                gossiped_states = []

                for agent in agents:
                    # Each agent calculates its new state based on the network state
                    gossiped_state = agent.gossip(network_state, W, return_state=True)
                    gossiped_states.append(gossiped_state)

                # Atomically update all agents with their new gossiped states
                for i, agent in enumerate(agents):
                    agent.apply_gossip_state(gossiped_states[i])

    return history
