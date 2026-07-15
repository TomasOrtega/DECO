# experiments/run_review_real_data_experiment.py
"""Review-motivated adaptive-baseline experiment on real datasets.

This companion to run_review_baseline_experiment.py runs the same DECO and
adaptive decentralized baselines on real, packaged scikit-learn datasets. The
original LIBSVM datasets remain supported by RealDataEnvironment when available
under data/, while the bundled datasets used here make Docker and CI sanity
checks independent of external download mirrors.
"""

import argparse
import csv
import os

import numpy as np

from src.deco.algorithms import run_simulation
from src.deco.environments import RealDataEnvironment
from src.deco.graph import create_gossip_matrix
from src.deco.online_baselines import (
    ONLINE_BASELINE_NAMES,
    make_online_baseline_config,
)
from src.deco.potentials import ExponentialPotential, KTPotential

DEFAULT_DATASETS = ("diabetes", "breast_cancer", "digits")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=100, help="online rounds")
    parser.add_argument("--N", type=int, default=20, help="number of agents")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[0], help="random seeds"
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0],
        help="base learning-rate scales for DOGD and adaptive baselines",
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--topology",
        default="cycle",
        choices=["cycle", "complete", "erdos_renyi"],
        help="network topology",
    )
    parser.add_argument("--results-dir", default="results")
    return parser.parse_args()


def make_algorithms(learning_rates):
    common = {"gossip": True, "disable_tqdm": True}
    algorithms = {
        "DECO-ii KT q=1": {
            "agent_type": "Deco",
            "potential": KTPotential(),
            "version": "ii",
            **common,
        },
        "DECO-i KT q=1": {
            "agent_type": "Deco",
            "potential": KTPotential(),
            "version": "i",
            **common,
        },
        "DECO-ii exp q=1": {
            "agent_type": "Deco",
            "potential": ExponentialPotential(),
            "version": "ii",
            **common,
        },
        "Centralized KT": {
            "agent_type": "Centralized",
            "potential": KTPotential(),
            "disable_tqdm": True,
        },
    }
    for name in ONLINE_BASELINE_NAMES:
        for learning_rate in learning_rates:
            label = f"{name} eta0={learning_rate:g}"
            algorithms[label] = make_online_baseline_config(
                name,
                learning_rate,
                **common,
            )
    return algorithms


def summarize(history):
    return {
        "final_cumulative_network_loss": float(np.sum(history["network_loss"])),
        "final_average_network_loss": float(np.mean(history["network_loss"])),
        "final_cumulative_local_loss": float(np.sum(history["local_loss"])),
        "communication_scalars": float(np.sum(history["communication_scalars"])),
    }


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    rows = []

    for seed in args.seeds:
        W = create_gossip_matrix(args.N, topology=args.topology, p=0.2, seed=seed)
        for dataset in args.datasets:
            for name, config in make_algorithms(args.learning_rates).items():
                env = RealDataEnvironment(args.N, dataset=dataset, seed=seed)
                history = run_simulation(
                    args.T, args.N, env.dim, env, W, config, env.u_star
                )
                row = {
                    "seed": seed,
                    "dataset": dataset,
                    "topology": args.topology,
                    "algorithm": name,
                    "initial_lr": config.get("lr", ""),
                    "n_samples": env.n_samples,
                    "dim": env.dim,
                }
                row.update(summarize(history))
                rows.append(row)
                print(row)

    out_path = os.path.join(args.results_dir, "review_real_data_summary.csv")
    fieldnames = [
        "seed",
        "dataset",
        "topology",
        "algorithm",
        "initial_lr",
        "n_samples",
        "dim",
        "final_cumulative_network_loss",
        "final_average_network_loss",
        "final_cumulative_local_loss",
        "communication_scalars",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
