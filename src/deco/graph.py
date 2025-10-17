# src/deco/graph.py
import networkx as nx
import numpy as np


def create_gossip_matrix(N, topology="cycle", p=0.2):
    """
    Creates a graph and its Metropolis-Hastings doubly stochastic gossip matrix.

    Args:
        N (int): Number of nodes.
        topology (str): Graph type ('cycle', 'complete', 'erdos_renyi').
        p (float): Connection probability for Erdos-Renyi graphs.

    Returns:
        np.ndarray: The N x N gossip matrix W.
    """
    if N <= 1:
        return np.array([[1.0]])

    if topology == "cycle":
        G = nx.cycle_graph(N)
    elif topology == "complete":
        G = nx.complete_graph(N)
    elif topology == "erdos_renyi":
        G = nx.erdos_renyi_graph(N, p)
        if not nx.is_connected(G):
            # Ensure the graph is connected by adding edges
            components = list(nx.connected_components(G))
            for i in range(len(components) - 1):
                G.add_edge(list(components[i])[0], list(components[i + 1])[0])
    else:
        raise ValueError(f"Unknown topology: {topology}")

    W = np.zeros((N, N))
    for i in range(N):
        neighbors = list(G.neighbors(i))
        for j in neighbors:
            W[i, j] = 1 / (max(G.degree(i), G.degree(j)) + 1)

    # Set diagonal elements for self-weights
    for i in range(N):
        W[i, i] = 1 - np.sum(W[i, :])

    return W
