import math
import unittest

from experiments.run_gossip_tradeoff_experiment import GOSSIP_SCHEDULES


class GossipScheduleTests(unittest.TestCase):
    def test_internal_index_matches_one_based_round_formulas(self):
        for zero_based_t in range(20):
            round_t = zero_based_t + 1
            self.assertEqual(GOSSIP_SCHEDULES["Constant (q=1)"](zero_based_t), 1)
            self.assertEqual(
                GOSSIP_SCHEDULES["Logarithmic (q=log(t))"](zero_based_t),
                math.ceil(math.log(round_t + 1)),
            )
            self.assertEqual(
                GOSSIP_SCHEDULES["Linear (q=0.1*t)"](zero_based_t),
                math.ceil(0.1 * round_t),
            )


if __name__ == "__main__":
    unittest.main()
