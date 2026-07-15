import unittest

import numpy as np

from src.deco.plotting import mask_large_losses


class PlottingTests(unittest.TestCase):
    def test_large_losses_are_masked_for_every_algorithm(self):
        losses = {
            "DOGD": [1.0, 6.0, 4.0],
            "D-Adam": [2.0, 10.0, 3.0],
            "DECO-i": [1.5, 1.5, 1.5],
        }

        masked = mask_large_losses(losses)

        np.testing.assert_allclose(masked["DOGD"][[0, 2]], [1.0, 4.0])
        np.testing.assert_allclose(masked["D-Adam"][[0, 2]], [2.0, 3.0])
        np.testing.assert_allclose(masked["DECO-i"], [1.5, 1.5, 1.5])
        self.assertTrue(np.isnan(masked["DOGD"][1]))
        self.assertTrue(np.isnan(masked["D-Adam"][1]))
        self.assertEqual(losses["DOGD"][1], 6.0)


if __name__ == "__main__":
    unittest.main()
