# experiments/run_connectivity_experiment.py
import numpy as np
import h5py
import os
from src.deco.algorithms import run_simulation
from src.deco.graph import create_gossip_matrix
from src.deco.environments import SyntheticRegression
from src.deco.potentials import KTPotential
from src.deco.utils import save_results_to_hdf5

CONFIG = {
    "T": 3000,
    "N": 20,
    "DIM": 10,
    "RESULTS_DIR": "results",
    "CONNECTIVITY_PROBS": [0.1, 0.3, 1.0],  # p for Erdos-Renyi
    "SEED": 0,
}

if __name__ == "__main__":
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    # fix random seed for reproducibility
    np.random.seed(CONFIG["SEED"])
    U_STAR = np.random.randn(CONFIG["DIM"])
    env = SyntheticRegression(CONFIG["N"], CONFIG["DIM"], U_STAR)

    centralized_config = {
        "agent_type": "Centralized",
        "potential": KTPotential(),
    }

    deco_config = {
        "agent_type": "Deco",
        "potential": KTPotential(),
        "version": "ii",
        "gossip": True,
    }

    all_results = {}

    # Run Centralized Oracle
    print("===== Running on Centralized Oracle =====")
    W_centralized = create_gossip_matrix(CONFIG["N"], topology="complete")
    results = run_simulation(
        CONFIG["T"],
        CONFIG["N"],
        CONFIG["DIM"],
        env,
        W_centralized,
        centralized_config,
        U_STAR,
    )
    all_results["Centralized"] = results

    # Run for different connectivity probabilities
    for p in CONFIG["CONNECTIVITY_PROBS"]:
        topo_name = f"ER (p={p})"
        print(f"===== Running on Topology: {topo_name} =====")
        W = create_gossip_matrix(CONFIG["N"], topology="erdos_renyi", p=p)

        results = run_simulation(
            CONFIG["T"], CONFIG["N"], CONFIG["DIM"], env, W, deco_config, U_STAR
        )
        all_results[topo_name] = results

    filepath = os.path.join(CONFIG["RESULTS_DIR"], "connectivity_results.h5")
    with h5py.File(filepath, "w") as f:
        save_results_to_hdf5(f, all_results)
    print(f"\nConnectivity experiment results saved to {filepath}")
