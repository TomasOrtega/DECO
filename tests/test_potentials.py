import unittest

import numpy as np

from src.deco.agents import DecoAgent
from src.deco.potentials import ExponentialPotential, KTPotential


class PotentialTests(unittest.TestCase):
    def test_agent_converts_zero_based_index_to_coin_betting_round(self):
        for version in ("i", "ii"):
            with self.subTest(version=version):
                potential = KTPotential()
                agent = DecoAgent(0, 1, potential, version=version)
                agent.predict(0)
                agent.update(np.array([-0.5]))
                agent.apply_gossip_state({"w": agent.hat_w, "G": agent.hat_G})

                prediction = agent.predict(1)
                if version == "i":
                    expected = potential.beta(2, agent.G) * agent.w
                else:
                    expected = potential.h(2, agent.G)

                np.testing.assert_allclose(prediction, expected)

    def test_exponential_potential_satisfies_first_round_boundary(self):
        potential = ExponentialPotential(epsilon=2.0)

        self.assertAlmostEqual(potential.F(1, 1.0), potential.epsilon)

    def test_exponential_beta_uses_radial_vector_extension(self):
        potential = ExponentialPotential()
        state = np.array([3.0, 4.0])

        expected = np.tanh(np.linalg.norm(state) / 6) * state / np.linalg.norm(state)

        np.testing.assert_allclose(potential.beta(6, state), expected)


if __name__ == "__main__":
    unittest.main()
