# experiments/tune_dgd.py
import numpy as np
import h5py
import os
from src.deco.algorithms import run_simulation
from src.deco.graph import create_gossip_matrix
from src.deco.environments import SyntheticRegression
from src.deco.utils import save_results_to_hdf5

CONFIG = {
    "T": 3000,
    "N": 20,
    "DIM": 10,
    "TOPOLOGY": "cycle",
    "RESULTS_DIR": "results",
    "SEED": 0,
}

LEARNING_RATES = np.logspace(-3, 3, num=25)

if __name__ == "__main__":
    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)
    # fix random seed for reproducibility
    np.random.seed(CONFIG["SEED"])
    U_STAR = np.random.randn(CONFIG["DIM"])
    env = SyntheticRegression(CONFIG["N"], CONFIG["DIM"], U_STAR)
    W = create_gossip_matrix(CONFIG["N"], topology=CONFIG["TOPOLOGY"])

    all_results = {}
    print(f"Tuning DGD over {len(LEARNING_RATES)} learning rates...")
    for lr in LEARNING_RATES:
        print(f"  Running DGD with initial_lr = {lr:.4f}")
        dgd_config = {
            "agent_type": "DGD",
            "lr": lr,
            "gossip": True,
        }
        results = run_simulation(
            CONFIG["T"], CONFIG["N"], CONFIG["DIM"], env, W, dgd_config, U_STAR
        )
        all_results[lr] = results

    filepath = os.path.join(CONFIG["RESULTS_DIR"], "dgd_tuning_results.h5")
    with h5py.File(filepath, "w") as f:
        save_results_to_hdf5(f, all_results)
    print(f"\nDGD tuning results saved to {filepath}")
