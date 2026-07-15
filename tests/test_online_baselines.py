import unittest

import numpy as np

from src.deco.algorithms import run_simulation
from src.deco.environments import SyntheticRegression
from src.deco.graph import create_gossip_matrix
from src.deco.online_baselines import (
    ADAPTIVE_BASELINE_NAMES,
    ONLINE_BASELINE_NAMES,
    make_online_baseline_config,
)
from src.deco.potentials import KTPotential


class OnlineBaselineTest(unittest.TestCase):
    def test_three_adaptive_baselines_are_exposed(self):
        self.assertEqual(
            ADAPTIVE_BASELINE_NAMES,
            ("D-AdaGrad", "D-RMSProp", "D-Adam"),
        )
        self.assertEqual(ONLINE_BASELINE_NAMES[0], "DOGD")

    def test_each_adaptive_baseline_runs_online_and_communicates_one_vector(self):
        n_agents = 4
        dimension = 3
        horizon = 5
        matrix = create_gossip_matrix(n_agents, topology="cycle")

        for name in ADAPTIVE_BASELINE_NAMES:
            with self.subTest(name=name):
                np.random.seed(7)
                comparator = np.random.randn(dimension)
                environment = SyntheticRegression(n_agents, dimension, comparator)
                config = make_online_baseline_config(
                    name,
                    learning_rate=0.1,
                    gossip=True,
                    disable_tqdm=True,
                )
                history = run_simulation(
                    horizon,
                    n_agents,
                    dimension,
                    environment,
                    matrix,
                    config,
                    comparator,
                )

                self.assertTrue(np.isfinite(history["network_loss"]).all())
                directed_cycle_edges = 2 * n_agents
                expected = horizon * directed_cycle_edges * dimension
                self.assertEqual(history["communication_scalars"].sum(), expected)

    def test_multiple_gossip_rounds_match_the_matrix_power(self):
        n_agents = 4
        dimension = 2
        horizon = 8
        matrix = create_gossip_matrix(n_agents, topology="cycle")
        powered_matrix = np.linalg.matrix_power(matrix, 3)
        comparator = np.array([0.5, -0.25])

        def run(mixing_matrix, rounds):
            np.random.seed(11)
            environment = SyntheticRegression(n_agents, dimension, comparator)
            config = make_online_baseline_config(
                "D-Adam",
                learning_rate=0.1,
                gossip=True,
                disable_tqdm=True,
                q_t=lambda _: rounds,
            )
            return run_simulation(
                horizon,
                n_agents,
                dimension,
                environment,
                mixing_matrix,
                config,
                comparator,
            )

        repeated = run(matrix, 3)
        powered = run(powered_matrix, 1)
        np.testing.assert_allclose(
            repeated["network_loss"],
            powered["network_loss"],
            rtol=1e-12,
            atol=1e-12,
        )

    def test_deco_multiple_gossip_rounds_match_the_matrix_power(self):
        n_agents = 4
        dimension = 2
        horizon = 8
        matrix = create_gossip_matrix(n_agents, topology="cycle")
        powered_matrix = np.linalg.matrix_power(matrix, 3)
        comparator = np.array([0.5, -0.25])

        def run(version, mixing_matrix, rounds):
            np.random.seed(11)
            environment = SyntheticRegression(n_agents, dimension, comparator)
            config = {
                "agent_type": "Deco",
                "potential": KTPotential(),
                "version": version,
                "gossip": True,
                "disable_tqdm": True,
                "q_t": lambda _: rounds,
            }
            return run_simulation(
                horizon,
                n_agents,
                dimension,
                environment,
                mixing_matrix,
                config,
                comparator,
            )

        for version in ("i", "ii"):
            with self.subTest(version=version):
                repeated = run(version, matrix, 3)
                powered = run(version, powered_matrix, 1)
                np.testing.assert_allclose(
                    repeated["network_loss"],
                    powered["network_loss"],
                    rtol=1e-12,
                    atol=1e-12,
                )


if __name__ == "__main__":
    unittest.main()
