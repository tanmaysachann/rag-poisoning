import unittest

import numpy as np

from poison.ppo import Transition, generalized_advantages


class PpoTests(unittest.TestCase):
    def test_terminal_and_truncated_rewards_do_not_bootstrap_across_episodes(self):
        template = dict(
            state=np.zeros(774), mask=np.ones((5, 3, 3), dtype=bool),
            action=(4, 0, 0), old_log_probability=0.0,
            value=0.0, proxy_value=0.0, proxy_reward=0.0,
        )
        rows = [
            Transition(**template, reward=2.0, done=True),
            Transition(**template, reward=7.0, done=True),
        ]
        advantages, returns = generalized_advantages(rows)
        np.testing.assert_allclose(advantages, [2.0, 7.0])
        np.testing.assert_allclose(returns, [2.0, 7.0])


if __name__ == "__main__":
    unittest.main()
