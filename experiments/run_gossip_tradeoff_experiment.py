# experiments/run_gossip_tradeoff_experiment.py
import numpy as np
import h5py
import os
import math
from src.deco.algorithms import run_simulation
from src.deco.graph import create_gossip_matrix
from src.deco.environments import SyntheticRegression
from src.deco.potentials import KTPotential
from src.deco.utils import save_results_to_hdf5

CONFIG = {
    "T": 3000,
    "N": 20,
    "DIM": 10,
    "TOPOLOGY": "cycle",
    "RESULTS_DIR": "results",
    "SEED": 0,
}

GOSSIP_SCHEDULES = {
    "Constant (q=1)": lambda t: 1,
    "Logarithmic (q=log(t))": lambda t: math.ceil(math.log(t + 1)),
    "Linear (q=0.1*t)": lambda t: math.ceil(0.1 * t),
}

if __name__ == "__main__":
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    # fix random seed for reproducibility
    np.random.seed(CONFIG["SEED"])
    U_STAR = np.random.randn(CONFIG["DIM"])
    env = SyntheticRegression(CONFIG["N"], CONFIG["DIM"], U_STAR)
    W = create_gossip_matrix(CONFIG["N"], topology=CONFIG["TOPOLOGY"])

    centralized_config = {
        "agent_type": "Centralized",
        "potential": KTPotential(),
    }

    all_results = {}

    # Run Centralized Oracle
    print("===== Running on Centralized Oracle =====")
    results = run_simulation(
        CONFIG["T"], CONFIG["N"], CONFIG["DIM"], env, W, centralized_config, U_STAR
    )
    all_results["Centralized"] = results

    # Run DECO with different gossip schedules
    for name, schedule_fn in GOSSIP_SCHEDULES.items():
        print(f"===== Running with Gossip Schedule: {name} =====")
        deco_config = {
            "agent_type": "Deco",
            "potential": KTPotential(),
            "version": "ii",
            "gossip": True,
            "q_t": schedule_fn,
        }
        results = run_simulation(
            CONFIG["T"], CONFIG["N"], CONFIG["DIM"], env, W, deco_config, U_STAR
        )
        all_results[name] = results

    filepath = os.path.join(CONFIG["RESULTS_DIR"], "gossip_tradeoff_results.h5")
    with h5py.File(filepath, "w") as f:
        save_results_to_hdf5(f, all_results)
    print(f"\nGossip tradeoff experiment results saved to {filepath}")
