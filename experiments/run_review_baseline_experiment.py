# experiments/run_review_baseline_experiment.py
"""Review-motivated communication-normalized adaptive baseline experiment.

This script adds the adaptive-gradient baselines requested by the reviewers and
reports both cumulative network loss and scalar communication cost. It is kept
synthetic and deterministic so it can be run quickly inside Docker, while the CLI
allows larger paper-scale runs by increasing --T and --seeds.
"""

import argparse
import csv
import os

import numpy as np

from src.deco.algorithms import run_simulation
from src.deco.environments import SyntheticRegression
from src.deco.graph import create_gossip_matrix
from src.deco.potentials import ExponentialPotential, KTPotential


DEFAULT_TOPOLOGIES = ("cycle", "erdos_renyi")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=200, help="online rounds")
    parser.add_argument("--N", type=int, default=20, help="number of agents")
    parser.add_argument("--dim", type=int, default=10, help="decision dimension")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0], help="random seeds")
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=list(DEFAULT_TOPOLOGIES),
        choices=["cycle", "complete", "erdos_renyi"],
        help="network topologies to evaluate",
    )
    parser.add_argument(
        "--heterogeneity-scale",
        type=float,
        default=4.0,
        help="agent feature-distribution heterogeneity",
    )
    parser.add_argument(
        "--results-dir", default="results", help="directory for the summary CSV"
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        default=True,
        help="disable per-run progress bars for batch/Docker logs",
    )
    return parser.parse_args()


def make_algorithms():
    common = {"gossip": True, "disable_tqdm": True}
    return {
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
        "DGD lr=1/sqrt(t)": {"agent_type": "DGD", "lr": 1.0, **common},
        "D-AdaGrad lr=0.3": {
            "agent_type": "AdaptiveDGD",
            "method": "adagrad",
            "lr": 0.3,
            **common,
        },
        "D-RMSProp lr=0.03": {
            "agent_type": "AdaptiveDGD",
            "method": "rmsprop",
            "lr": 0.03,
            "beta2": 0.99,
            **common,
        },
        "D-Adam lr=0.03": {
            "agent_type": "AdaptiveDGD",
            "method": "adam",
            "lr": 0.03,
            **common,
        },
        "D-AdamW lr=0.03": {
            "agent_type": "AdaptiveDGD",
            "method": "adamw",
            "lr": 0.03,
            "weight_decay": 1e-3,
            **common,
        },
        "D-Momentum lr=0.1": {
            "agent_type": "AdaptiveDGD",
            "method": "momentum",
            "lr": 0.1,
            **common,
        },
        "D-Nesterov lr=0.1": {
            "agent_type": "AdaptiveDGD",
            "method": "nesterov",
            "lr": 0.1,
            **common,
        },
        "Centralized KT": {
            "agent_type": "Centralized",
            "potential": KTPotential(),
            "disable_tqdm": True,
        },
    }


def summarize(history):
    return {
        "final_cumulative_network_loss": float(np.sum(history["network_loss"])),
        "final_average_network_loss": float(np.mean(history["network_loss"])),
        "final_cumulative_local_loss": float(np.sum(history["local_loss"])),
        "communication_scalars": float(np.sum(history["communication_scalars"])),
    }


def write_summary(rows, out_path):
    fieldnames = [
        "seed",
        "topology",
        "algorithm",
        "final_cumulative_network_loss",
        "final_average_network_loss",
        "final_cumulative_local_loss",
        "communication_scalars",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    rows = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        u_star = rng.standard_normal(args.dim)
        for topology in args.topologies:
            # Keep Erdos-Renyi reproducible without changing existing graph APIs.
            np.random.seed(seed)
            W = create_gossip_matrix(args.N, topology=topology, p=0.2)
            for name, algo_config in make_algorithms().items():
                np.random.seed(seed)
                env = SyntheticRegression(
                    args.N,
                    args.dim,
                    u_star,
                    heterogeneity_scale=args.heterogeneity_scale,
                )
                history = run_simulation(
                    args.T, args.N, args.dim, env, W, algo_config, u_star
                )
                row = {"seed": seed, "topology": topology, "algorithm": name}
                row.update(summarize(history))
                rows.append(row)
                print(row)

    out_path = os.path.join(args.results_dir, "review_baseline_summary.csv")
    write_summary(rows, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
