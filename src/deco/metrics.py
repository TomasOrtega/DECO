# src/deco/metrics.py
import numpy as np


def calculate_losses(local_losses, decisions, agent_features, agent_labels):
    """
    Calculates the average local loss and average network loss for a single timestep.

    Args:
        local_losses (np.ndarray): The loss incurred by each agent for the step.
        decisions (list): The decision made by each agent, [x_1, ...].
        agent_features (list): The feature vector seen by each agent.
        agent_labels (list): The label seen by each agent.

    Returns:
        (float, float): A tuple of (average_local_loss, average_network_loss).
    """
    # 1. Average Local Loss
    avg_local_loss = np.mean(local_losses)

    # 2. Average Network Loss
    N = len(decisions)

    # The global loss l(x) is the average loss over all agents' data.
    def global_loss(x):
        total_loss = 0
        for i in range(N):
            pred = np.dot(agent_features[i], x)
            total_loss += np.abs(pred - agent_labels[i])
        return total_loss / N

    # Average the global loss for each agent's decision
    avg_network_loss = np.mean([global_loss(x_n) for x_n in decisions])

    return avg_local_loss, avg_network_loss
