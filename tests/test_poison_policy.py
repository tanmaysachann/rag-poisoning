import unittest

import numpy as np
import torch

from poison.actions import OPERATIONS
from poison.policy import FactoredActorCritic


class PolicyTests(unittest.TestCase):
    def test_factored_log_probability_and_mask(self):
        torch.manual_seed(3)
        policy = FactoredActorCritic()
        mask = np.zeros((len(OPERATIONS), 3, 3), dtype=np.bool_)
        mask[OPERATIONS.index("INSERT"), 0, 1] = True
        action, log_probability, value, proxy_value = policy.act(np.zeros(774), mask)
        self.assertEqual((action.operation, action.position, action.payload), ("INSERT", 0, 1))
        self.assertAlmostEqual(log_probability, 0.0, places=5)
        self.assertTrue(np.isfinite(value))
        self.assertTrue(np.isfinite(proxy_value))


if __name__ == "__main__":
    unittest.main()
