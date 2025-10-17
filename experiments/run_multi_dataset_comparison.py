# experiments/run_multi_dataset_comparison.py
import numpy as np
import h5py
import os
import time
from tqdm import tqdm

from src.deco.algorithms import run_simulation
from src.deco.graph import create_gossip_matrix
from src.deco.environments import RealDataEnvironment
from src.deco.potentials import ExponentialPotential, KTPotential
from src.deco.utils import save_results_to_hdf5

# Configuration
CONFIG = {
    "N": 20,  # Number of agents
    "TOPOLOGY": "cycle",
    "RESULTS_DIR": "results",
}

# Add the learning rates for DGD tuning
LEARNING_RATES_DGD = np.logspace(-3, 7, num=100)

# Datasets to compare
DATASETS_TO_TEST = [
    "space_ga",
    "cpusmall",
    "cadata",
    "abalone",
]

ALGORITHMS = {
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
    "Centralized": {"agent_type": "Centralized", "potential": KTPotential()},
}


def run_single_dataset_experiment(dataset_name):
    """Run all algorithms on a single dataset, including DGD tuning."""
    print(f"Starting experiments on {dataset_name}...")

    try:
        env = RealDataEnvironment(CONFIG["N"], dataset=dataset_name)
    except ValueError as e:
        print(f"Error loading dataset {dataset_name}: {e}")
        return None

    T = int(env.n_samples // CONFIG["N"])
    print(f"  Using T={T} iterations (floor({env.n_samples} / {CONFIG['N']}))")

    W = create_gossip_matrix(CONFIG["N"], topology=CONFIG["TOPOLOGY"])
    U_STAR = env.u_star

    dataset_results = {}

    # Run parameter-free algorithms
    for algo_name, algo_config in ALGORITHMS.items():
        print(f"  Running {algo_name} on {dataset_name}...")
        start_time = time.time()
        results = run_simulation(T, CONFIG["N"], env.dim, env, W, algo_config, U_STAR)
        elapsed = time.time() - start_time
        dataset_results[algo_name] = results
        print(f"    Completed in {elapsed:.1f}s")

    # Run DGD tuning sweep
    print(
        f"  Tuning DGD on {dataset_name} over {len(LEARNING_RATES_DGD)} "
        f"learning rates..."
    )
    dgd_results = {}
    for lr in tqdm(LEARNING_RATES_DGD, desc="    DGD tuning", leave=False):
        dgd_config = {"agent_type": "DGD", "lr": lr, "gossip": True}
        results = run_simulation(T, CONFIG["N"], env.dim, env, W, dgd_config, U_STAR)
        dgd_results[lr] = results

    dataset_results["DGD_tune"] = dgd_results

    return {
        "dataset_name": dataset_name,
        "dataset_info": {
            "task_type": env.task_type,
            "n_features": env.dim,
            "n_samples": env.n_samples,
            "T": T,
        },
        "results": dataset_results,
    }


def main():
    """Main execution function."""
    print("Multi-Dataset Comparison Experiment")
    print(f"Testing {len(DATASETS_TO_TEST)} datasets with {len(ALGORITHMS)} algorithms")
    print(f"Configuration: N={CONFIG['N']}, topology={CONFIG['TOPOLOGY']}")
    print("T (iterations) will be calculated per dataset as floor(n_samples / N)")
    print()

    os.makedirs(CONFIG["RESULTS_DIR"], exist_ok=True)

    all_results = []
    start_time = time.time()

    for dataset_name in DATASETS_TO_TEST:
        result = run_single_dataset_experiment(dataset_name)
        if result:
            all_results.append(result)

    total_time = time.time() - start_time

    # Create a serializable version of the ALGORITHMS dict for metadata
    serializable_algorithms = {
        name: {k: str(v) for k, v in conf.items()} for name, conf in ALGORITHMS.items()
    }

    # Save comprehensive results
    output_data = {
        "config": CONFIG,
        "algorithms": serializable_algorithms,
        "datasets_tested": DATASETS_TO_TEST,
        "results": all_results,
        "total_runtime": total_time,
    }

    filepath = os.path.join(CONFIG["RESULTS_DIR"], "multi_dataset_comparison.h5")
    with h5py.File(filepath, "w") as f:
        save_results_to_hdf5(f, output_data)

    print(f"\nAll results saved to {filepath}")
    print(f"Total runtime: {total_time:.1f} seconds")


if __name__ == "__main__":
    main()
