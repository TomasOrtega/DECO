# experiments/tune_dgd.py
import numpy as np
import h5py
import os

from src.deco.algorithms import run_simulation
from src.deco.environments import SyntheticRegression
from src.deco.graph import create_gossip_matrix
from src.deco.online_baselines import (
    LEARNING_RATES,
    ONLINE_BASELINE_NAMES,
    make_online_baseline_config,
)
from src.deco.utils import save_results_to_hdf5

CONFIG = {
    "T": 3000,
    "N": 20,
    "DIM": 10,
    "TOPOLOGY": "cycle",
    "RESULTS_DIR": "results",
    "SEED": 0,
}


def make_environment(u_star):
    """Recreate the same online stream for every method and learning rate."""
    np.random.seed(CONFIG["SEED"])
    return SyntheticRegression(CONFIG["N"], CONFIG["DIM"], u_star)

if __name__ == "__main__":
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    rng = np.random.default_rng(CONFIG["SEED"])
    U_STAR = rng.standard_normal(CONFIG["DIM"])
    W = create_gossip_matrix(CONFIG["N"], topology=CONFIG["TOPOLOGY"])

    all_results = {}
    for name in ONLINE_BASELINE_NAMES:
        print(f"Tuning {name} over {len(LEARNING_RATES)} learning rates...")
        rate_results = {}
        for lr in LEARNING_RATES:
            print(f"  Running {name} with eta_0 = {lr:.4g}")
            config = make_online_baseline_config(
                name,
                lr,
                gossip=True,
                disable_tqdm=True,
            )
            rate_results[lr] = run_simulation(
                CONFIG["T"],
                CONFIG["N"],
                CONFIG["DIM"],
                make_environment(U_STAR),
                W,
                config,
                U_STAR,
            )
        all_results[name] = rate_results

    filepath = os.path.join(
        CONFIG["RESULTS_DIR"], "online_baseline_tuning_results.h5"
    )
    with h5py.File(filepath, "w") as f:
        save_results_to_hdf5(f, all_results)
    print(f"\nOnline baseline tuning results saved to {filepath}")
