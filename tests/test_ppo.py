import unittest

import numpy as np

from poison.actions import EditAction
from poison.edit_cache import EditEffectCache
from poison.ppo import Transition, collect_episode, generalized_advantages


class PpoTests(unittest.TestCase):
    def test_edit_cache_shapes_training_reward_only_after_observed_edit(self):
        class FixtureEnv:
            def reset(self, *, seed):
                self.steps = 0
                return np.zeros(774, dtype=np.float32), {"action_mask": np.ones((5, 3, 3), dtype=bool)}

            def step(self, action):
                self.steps += 1
                done = self.steps == 2
                return (np.full(774, self.steps, dtype=np.float32), 0.4,
                        done, False, {"valid": True, "reward_components": {"total": 0.4},
                                      "action_mask": np.ones((5, 3, 3), dtype=bool),
                                      "terminal": {"attack_success": False}})

        class FixturePolicy:
            def __init__(self):
                self.steps = 0

            def act(self, state, mask):
                self.steps += 1
                return (EditAction("INSERT", 0, 1) if self.steps % 2 else EditAction("STOP"),
                        0.0, 0.0, 0.0)

        cache = EditEffectCache(state_dim=774)
        policy = FixturePolicy()
        first, _ = collect_episode(FixtureEnv(), policy, 7, edit_cache=cache, cache_weight=0.5)
        second, _ = collect_episode(FixtureEnv(), policy, 7, edit_cache=cache, cache_weight=0.5)
        self.assertEqual(len(cache.rows), 2)
        self.assertAlmostEqual(first[0].reward, 0.4)
        self.assertAlmostEqual(second[0].reward, 0.6)
        self.assertAlmostEqual(second[0].proxy_reward, 0.6)
        self.assertAlmostEqual(second[1].reward, 0.4)

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
