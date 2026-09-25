"""Baseline edits must preserve the seed passage and avoid the true alias."""

from __future__ import annotations

import unittest

from attack.baselines import choose_wrong_answer, insert_false_answer, substitute_answer


class AttackBaselineTests(unittest.TestCase):
    def test_single_answer_substitution_preserves_surrounding_text(self) -> None:
        seed = "The Eiffel Tower is located in Paris, France. The structure opened in 1889."
        edited = substitute_answer(seed, ["Paris"], "London")
        self.assertIn("London, France", edited)
        self.assertIn("The structure opened in 1889.", edited)
        self.assertNotIn("Paris", edited)
        self.assertEqual(
            substitute_answer("The city is New-York City.", ["New York"], "London"),
            "The city is London City.",
        )

    def test_insert_positions_preserve_seed_and_target_wrong_alias(self) -> None:
        seed = "A clean first sentence. A clean second sentence."
        wrong = choose_wrong_answer(["Paris"], ["Paris", "London", "Rome"], seed=5)
        self.assertNotEqual(wrong, "Paris")
        for position in ("start", "middle", "end"):
            with self.subTest(position=position):
                edited = insert_false_answer(seed, "Where is the tower?", wrong, position)
                self.assertIn("A clean first sentence.", edited)
                self.assertIn("A clean second sentence.", edited)
                self.assertIn(wrong, edited)


if __name__ == "__main__":
    unittest.main()
