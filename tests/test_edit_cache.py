import unittest

import numpy as np

from poison.actions import EditAction
from poison.edit_cache import EditEffectCache


class EditCacheTests(unittest.TestCase):
    def test_neighbors_are_action_specific_and_predict_delta(self):
        cache = EditEffectCache(3)
        before = np.array([0.0, 1.0, 0.0])
        after = np.array([1.0, 1.0, 0.0])
        insert = EditAction("INSERT", 0, 1)
        cache.add(before, insert, after, 1.25)
        cache.add(before, EditAction("STOP"), before, 0.0)
        prediction = cache.predict(before, insert)
        self.assertEqual(prediction["neighbors"], 1)
        np.testing.assert_allclose(prediction["next_state"], after)
        self.assertEqual(prediction["expected_reward"], 1.25)
        self.assertIsNone(cache.predict(before, EditAction("DELETE")))
        with self.assertRaises(ValueError):
            cache.add(np.zeros(2), insert, after, 1.0)


if __name__ == "__main__":
    unittest.main()
