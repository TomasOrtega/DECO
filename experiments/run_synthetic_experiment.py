# experiments/run_synthetic_experiment.py
import numpy as np
import h5py
import os

from src.deco.algorithms import run_simulation
from src.deco.graph import create_gossip_matrix
from src.deco.environments import SyntheticRegression
from src.deco.potentials import ExponentialPotential, KTPotential
from src.deco.utils import save_results_to_hdf5

# --- Experiment Configuration ---
CONFIG = {
    "T": 3000,
    "N": 20,
    "DIM": 10,
    "TOPOLOGIES": ["cycle", "complete"],
    "RESULTS_DIR": "results",
    "SEED": 0,
}

if __name__ == "__main__":
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    U_STAR = np.random.randn(CONFIG["DIM"])
    # fix random seed for reproducibility
    np.random.seed(CONFIG["SEED"])

    for topo in CONFIG["TOPOLOGIES"]:
        print(f"===== Running on Topology: {topo.upper()} =====")
        W = create_gossip_matrix(CONFIG["N"], topology=topo)
        env = SyntheticRegression(CONFIG["N"], CONFIG["DIM"], U_STAR)

        algorithms = {
            "DECO-ii (exp)": {
                "agent_type": "Deco",
                "potential": ExponentialPotential(),
                "version": "ii",
                "gossip": True,
            },
            "DECO-ii (KT)": {
                "agent_type": "Deco",
                "potential": KTPotential(),
                "version": "ii",
                "gossip": True,
            },
            "DECO-i (exp)": {
                "agent_type": "Deco",
                "potential": ExponentialPotential(),
                "version": "i",
                "gossip": True,
            },
            "DECO-i (KT)": {
                "agent_type": "Deco",
                "potential": KTPotential(),
                "version": "i",
                "gossip": True,
            },
            "DGD (initial_lr=0.1)": {"agent_type": "DGD", "lr": 0.1, "gossip": True},
            "DGD (initial_lr=1.0)": {"agent_type": "DGD", "lr": 1.0, "gossip": True},
            "DGD (initial_lr=10.0)": {"agent_type": "DGD", "lr": 10.0, "gossip": True},
            "Centralized": {
                "agent_type": "Centralized",
                "potential": KTPotential(),
            },
        }

        all_results = {}
        for name, algo_config in algorithms.items():
            print(f"--- Running Algorithm: {name} ---")
            results = run_simulation(
                CONFIG["T"], CONFIG["N"], CONFIG["DIM"], env, W, algo_config, U_STAR
            )
            all_results[name] = results

        filepath = os.path.join(CONFIG["RESULTS_DIR"], f"synthetic_results_{topo}.h5")
        with h5py.File(filepath, "w") as f:
            save_results_to_hdf5(f, all_results)
        print(f"Results for {topo} topology saved to {filepath}\n")
