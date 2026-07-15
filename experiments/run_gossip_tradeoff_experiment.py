# experiments/run_gossip_tradeoff_experiment.py
import math
import os

import h5py
import numpy as np

from src.deco.algorithms import run_simulation
from src.deco.environments import SyntheticRegression
from src.deco.graph import create_gossip_matrix
from src.deco.online_baselines import (
    ADAPTIVE_BASELINE_NAMES,
    REFERENCE_LEARNING_RATE,
    make_online_baseline_config,
)
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
    "Logarithmic (q=log(t))": lambda t: math.ceil(math.log(t + 2)),
    "Linear (q=0.1*t)": lambda t: math.ceil(0.1 * (t + 1)),
}


def make_environment(u_star):
    np.random.seed(CONFIG["SEED"])
    return SyntheticRegression(CONFIG["N"], CONFIG["DIM"], u_star)


if __name__ == "__main__":
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    rng = np.random.default_rng(CONFIG["SEED"])
    U_STAR = rng.standard_normal(CONFIG["DIM"])
    W = create_gossip_matrix(CONFIG["N"], topology=CONFIG["TOPOLOGY"])

    centralized_config = {
        "agent_type": "Centralized",
        "potential": KTPotential(),
    }

    all_results = {}

    # Run Centralized Oracle
    print("===== Running on Centralized Oracle =====")
    results = run_simulation(
        CONFIG["T"],
        CONFIG["N"],
        CONFIG["DIM"],
        make_environment(U_STAR),
        W,
        centralized_config,
        U_STAR,
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
            CONFIG["T"],
            CONFIG["N"],
            CONFIG["DIM"],
            make_environment(U_STAR),
            W,
            deco_config,
            U_STAR,
        )
        all_results[name] = results

    for name in ADAPTIVE_BASELINE_NAMES:
        print(f"===== Running {name} with q(t)=1 =====")
        config = make_online_baseline_config(
            name,
            REFERENCE_LEARNING_RATE,
            gossip=True,
            disable_tqdm=True,
            q_t=lambda t: 1,
        )
        label = f"{name} (q=1, eta0={REFERENCE_LEARNING_RATE:g})"
        all_results[label] = run_simulation(
            CONFIG["T"],
            CONFIG["N"],
            CONFIG["DIM"],
            make_environment(U_STAR),
            W,
            config,
            U_STAR,
        )

    filepath = os.path.join(CONFIG["RESULTS_DIR"], "gossip_tradeoff_results.h5")
    with h5py.File(filepath, "w") as f:
        save_results_to_hdf5(f, all_results)
    print(f"\nGossip tradeoff experiment results saved to {filepath}")
